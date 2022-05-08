#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "cuda_runtime_api.h"
#include "cuda_utils.h"
#include "common.hpp"
#include "logging.h"
// #include "utils.h"



#define USE_FP16
#define DEVICE 0 // GPU id
#define BATCH_SIZE 1

static const int INPUT_C_UFLD = 3;
static const int INPUT_H_UFLD = 288;
static const int INPUT_W_UFLD = 800;
static const int OUTPUT_C_UFLD = 101;
static const int OUTPUT_H_UFLD = 56;
static const int OUTPUT_W_UFLD = 4;
static const int OUTPUT_SIZE_UFLD = OUTPUT_C_UFLD * OUTPUT_H_UFLD * OUTPUT_W_UFLD;
const char *INPUT_BLOB_NAME_UFLD = "data";
const char *OUTPUT_BLOB_NAME_UFLD = "prob";
static Logger gLogger;

typedef struct 
{
    float *data;
    float *prob;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *exe_context;
    void* buffers[2];
    cudaStream_t cuda_stream;
    int inputIndex;
    int outputIndex;
}UfldTRTContext;


static void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME_UFLD);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME_UFLD);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C_UFLD * INPUT_H_UFLD * INPUT_W_UFLD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE_UFLD * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C_UFLD * INPUT_H_UFLD * INPUT_W_UFLD * sizeof(float),
          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE_UFLD * sizeof(float),
          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

}

std::vector<float> prepareImage(cv::Mat & img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W_UFLD, INPUT_H_UFLD));

    cv::Mat img_float;

    resized.convertTo(img_float, CV_32FC3, 1. / 255.);

    // HWC TO CHW
    std::vector<cv::Mat> input_channels(INPUT_C_UFLD);
    cv::split(img_float, input_channels);

    // normalize
    std::vector<float> result(INPUT_H_UFLD * INPUT_W_UFLD * INPUT_C_UFLD);
    auto data = result.data();
    int channelLength = INPUT_H_UFLD * INPUT_W_UFLD;
    static float mean[]= {0.485, 0.456, 0.406};
    static float std[] = {0.229, 0.224, 0.225};
    for (int i = 0; i < INPUT_C_UFLD; ++i) {
        cv::Mat normed_channel = (input_channels[i] - mean[i]) / std[i];
        memcpy(data, normed_channel.data, channelLength * sizeof(float));
        data += channelLength;
    }

    return result;
}

/* (101,56,4), add softmax on 101_axis and calculate Expect */
void softmax_mul(float* x, float* y, int rows, int cols, int chan)
{
    for(int i = 0, wh = rows * cols; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            float sum = 0.0;
            float expect = 0.0;
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = exp(x[k * wh + i * cols + j]);
                sum += x[k * wh + i * cols + j];
            }
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] /= sum;
            }
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = x[k * wh + i * cols + j] * (k + 1);
                expect += x[k * wh + i * cols + j];
            }
            y[i * cols + j] = expect;
        }
    }
}

/* (101,56,4), calculate max index on 101_axis */
void argmax(float* x, float* y, int rows, int cols, int chan)
{
    for(int i = 0,wh = rows * cols; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            int max = -10000000;
            int max_ind = -1;
            for(int k = 0; k < chan; k++)
            {
                if(x[k * wh + i * cols + j] > max)
                {
                    max = x[k * wh + i * cols + j];
                    max_ind = k;
                }
            }
            y[i * cols + j] = max_ind;
        }
    }
}

void * ufld_trt_create(const char * engine_name)
{
    cudaSetDevice(DEVICE);
    size_t size = 0;
    char *trtModelStream = NULL;
    UfldTRTContext * trt_ctx = NULL;
    trt_ctx = new UfldTRTContext();

    std::ifstream file(engine_name, std::ios::binary);
    printf("Ufld_trt_create  ... \n");
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }else
        return NULL;

    trt_ctx->data = new float[BATCH_SIZE * INPUT_C_UFLD * INPUT_H_UFLD * INPUT_W_UFLD];
    trt_ctx->prob = new float[BATCH_SIZE * OUTPUT_SIZE_UFLD];

    trt_ctx->runtime = createInferRuntime(gLogger);
    assert(trt_ctx->runtime != nullptr);
    printf("Ufld_trt_create  cuda engine... \n");
    trt_ctx->engine = trt_ctx->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(trt_ctx->engine != nullptr);
    trt_ctx->exe_context = trt_ctx->engine->createExecutionContext();

    delete[] trtModelStream;

    return (void *)trt_ctx;
}

int ufld_trt_det(void *h, cv::Mat &img_in,cv::Mat& im_color)
{
    UfldTRTContext *trt_ctx;
    int i;
    int delay_preprocess;
    int delay_infer;
 
    trt_ctx = (UfldTRTContext *)h;
    
    int vis_h = 720;
    int vis_w = 1280;
    int col_sample_w = 8;

    cv::Mat vis;
    cv::Mat img = cv::imread("/workspace/yolo_ros_trt_docker/src/yolo_deepsort/ufld/samples/Strada_Provinciale_BS_510_Sebina_Orientale.jpg");
    cv::resize(img, vis, cv::Size(vis_w, vis_h));
    std::cout << "Resizing Finished" << std::endl;
    std::vector<float> result(INPUT_C_UFLD * INPUT_W_UFLD * INPUT_H_UFLD);
    result = prepareImage(img);
    std::cout << "Image Input is " << result[0] <<std::endl;
    memcpy(trt_ctx->data, &result[0], INPUT_C_UFLD * INPUT_W_UFLD * INPUT_H_UFLD * sizeof(float));
    std::cout << "Input preparation finished" << std::endl;
    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*trt_ctx->exe_context, trt_ctx->data, trt_ctx->prob, BATCH_SIZE); //prob: size (101, 56, 4)
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time is "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " ms" << std::endl;

    std::vector<int> tusimple_row_anchor
        { 64,  68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284 };

    float max_ind[BATCH_SIZE * OUTPUT_H_UFLD * OUTPUT_W_UFLD];
    float prob_reverse[BATCH_SIZE * OUTPUT_SIZE_UFLD];

    float expect[BATCH_SIZE * OUTPUT_H_UFLD * OUTPUT_W_UFLD];
    for (int k = 0, wh = OUTPUT_W_UFLD * OUTPUT_H_UFLD; k < OUTPUT_C_UFLD; k++)
    {
        for(int j = 0; j < OUTPUT_H_UFLD; j ++)
        {
            for(int l = 0; l < OUTPUT_W_UFLD; l++)
            {
                prob_reverse[k * wh + (OUTPUT_H_UFLD - 1 - j) * OUTPUT_W_UFLD + l] =
                    trt_ctx->prob[k * wh + j * OUTPUT_W_UFLD + l];
            }
        }
    }

    argmax(prob_reverse, max_ind, OUTPUT_H_UFLD, OUTPUT_W_UFLD, OUTPUT_C_UFLD);
    /* calculate softmax and Expect */
    softmax_mul(prob_reverse, expect, OUTPUT_H_UFLD, OUTPUT_W_UFLD, OUTPUT_C_UFLD);
    for(int k = 0; k < OUTPUT_H_UFLD; k++) {
        for(int j = 0; j < OUTPUT_W_UFLD; j++) {
            max_ind[k * OUTPUT_W_UFLD + j] == 100 ? expect[k * OUTPUT_W_UFLD + j] = 0 :
                expect[k * OUTPUT_W_UFLD + j] = expect[k * OUTPUT_W_UFLD + j];
        }
    }
    std::vector<int> i_ind;
    for(int k = 0; k < OUTPUT_W_UFLD; k++) {
        int ii = 0;
        for(int g = 0; g < OUTPUT_H_UFLD; g++) {
            if(expect[g * OUTPUT_W_UFLD + k] != 0)
                ii++;
        }
        if(ii > 2) {
            i_ind.push_back(k);
        }
    }
    for(int k = 0; k < OUTPUT_H_UFLD; k++) {
        for(int ll = 0; ll < i_ind.size(); ll++) {
            if(expect[OUTPUT_W_UFLD * k + i_ind[ll]] > 0) {
                cv::Point pp =
                    { int(expect[OUTPUT_W_UFLD * k + i_ind[ll]] * col_sample_w * vis_w / INPUT_W_UFLD) - 1,
                        int( vis_h * tusimple_row_anchor[OUTPUT_H_UFLD - 1 - k] / INPUT_H_UFLD) - 1 };
                cv::circle(vis, pp, 8, CV_RGB(0, 255 ,0), 2);
            }
        }
    }
    // cv::imshow("lane_vis",vis);
    // cv::waitKey(1);
    // cv::imwrite("lane_vis",vis);

}

void ufld_trt_destroy(void *h)
{
    UfldTRTContext *trt_ctx;
 
    trt_ctx = (UfldTRTContext *)h;
 
    // Release stream and buffers
    cudaStreamDestroy(trt_ctx->cuda_stream);
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]));
    //cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]);
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->outputIndex]));
    //cudaFree(trt_ctx->buffers[trt_ctx->outputIndex])
    // Destroy the engine
    trt_ctx->exe_context->destroy();
    trt_ctx->engine->destroy();
    trt_ctx->runtime->destroy();
 
    delete trt_ctx->data;
    delete trt_ctx->prob;
 
    delete trt_ctx;
}
