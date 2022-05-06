#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "cuda_runtime_api.h"
#include "cuda_utils.h"
#include "common_hrnet.hpp"
#include "logging.h"
// #include "utils.h"


static Logger gLogger;
#define USE_FP32
#define DEVICE 0 // GPU id
#define BATCH_SIZE 1

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "output";
static const int INPUT_H = 512;
static const int INPUT_W = 1024;
static const int NUM_CLASSES = 19;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W;

typedef struct 
{
    float *data;
    int *prob;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *exe_context;
    void* buffers[2];
    cudaStream_t cuda_stream;
    int inputIndex;
    int outputIndex;
}HrnetTRTContext;

// static void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, int* output, int batchSize) {
//     // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//     CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//     //cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
//     context.enqueue(batchSize, buffers, stream, nullptr);
//     CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream));
//     //cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
//     cudaStreamSynchronize(stream);
// }

static void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, int batchSize)
{
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
}

void * hrnet_trt_create(const char * engine_name)
{
    cudaSetDevice(DEVICE);
    size_t size = 0;
    char *trtModelStream = NULL;
    HrnetTRTContext * trt_ctx = NULL;
    trt_ctx = new HrnetTRTContext();

    std::ifstream file(engine_name, std::ios::binary);
    printf("hrnet_trt_create  ... \n");
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

    cudaSetDeviceFlags(cudaDeviceMapHost);
    // trt_ctx->data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    // trt_ctx->prob = new int[BATCH_SIZE * OUTPUT_SIZE];
    // Create GPU buffers on device
    printf("hrnet_seg_trt_create  buffer ... \n");
    CHECK(cudaHostAlloc((void **)&trt_ctx->data, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&trt_ctx->prob, BATCH_SIZE * OUTPUT_SIZE * sizeof(int), cudaHostAllocMapped));

    trt_ctx->runtime = createInferRuntime(gLogger);
    assert(trt_ctx->runtime != nullptr);
    printf("hrnet_seg_trt_create  cuda engine... \n");
    trt_ctx->engine = trt_ctx->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(trt_ctx->engine != nullptr);
    trt_ctx->exe_context = trt_ctx->engine->createExecutionContext();
    
    delete[] trtModelStream;

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    trt_ctx->inputIndex = trt_ctx->engine->getBindingIndex(INPUT_BLOB_NAME);
    trt_ctx->outputIndex = trt_ctx->engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(trt_ctx->inputIndex == 0);
    assert(trt_ctx->outputIndex == 1);

    // CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));   

    printf("hrnet_seg_trt_create  stream ... \n");
    CHECK(cudaStreamCreate(&trt_ctx->cuda_stream));
    printf("hrnet_seg_trt_create  done ... \n");
    return (void *)trt_ctx;
}

int hrnet_trt_seg(void *h, cv::Mat &img,cv::Mat& outimg)
{
    HrnetTRTContext *trt_ctx;
    int i;
    int delay_preprocess;
    int delay_infer;
 
    trt_ctx = (HrnetTRTContext *)h;
    
    cv::Mat pr_img;
    cv::Mat img_BGR = cv::imread("/workspace/tensorrtx/ufld/samples/Strada_Provinciale_BS_510_Sebina_Orientale.jpg", 1); // BGR
    cv::cvtColor(img_BGR, pr_img, cv::COLOR_BGR2RGB);
    cv::resize(pr_img, pr_img, cv::Size(INPUT_W, INPUT_H));
    pr_img.convertTo(pr_img, CV_32FC3);
    if (!pr_img.isContinuous())
    {
        pr_img = pr_img.clone();
    }
    std::memcpy(trt_ctx->data, pr_img.data, BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(int));

    cudaHostGetDevicePointer((void **)&trt_ctx->buffers[trt_ctx->inputIndex], (void *)trt_ctx->data, 0);  // buffers[inputIndex]-->data
    cudaHostGetDevicePointer((void **)&trt_ctx->buffers[trt_ctx->outputIndex], (void *)trt_ctx->prob, 0); // buffers[outputIndex] --> prob

    // cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);

    // // letterbox BGR to RGB
    // i = 0;
    // for (int row = 0; row < INPUT_H; ++row) {
    //     uchar* uc_pixel = pr_img.data + row * pr_img.step;
    //     for (int col = 0; col < INPUT_W; ++col) {
    //         trt_ctx->data[i] = (float)uc_pixel[2] / 255.0;
    //         trt_ctx->data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
    //         trt_ctx->data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
    //         uc_pixel += 3;
    //         ++i;
    //     }
    // }
    cv::imwrite("pri_map.png", pr_img);

    auto start = std::chrono::system_clock::now();
    doInference(*trt_ctx->exe_context, trt_ctx->cuda_stream, trt_ctx->buffers, BATCH_SIZE);
    // doInference(*trt_ctx->exe_context, trt_ctx->cuda_stream, trt_ctx->buffers, trt_ctx->data, trt_ctx->prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
 
    std::cout << "delay_infer: " << delay_infer << " ms" << std::endl;
    

    outimg = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);
    for (int row = 0; row < INPUT_H; ++row)
    {
        uchar *uc_pixel = outimg.data + row * outimg.step;
        for (int col = 0; col < INPUT_W; ++col)
        {
            uc_pixel[col] = (uchar)trt_ctx->prob[row * INPUT_W + col];
        }
    }
    cv::imwrite("out_map.png", outimg);
    cv::Mat im_color;
    cv::cvtColor(outimg, im_color, cv::COLOR_GRAY2RGB);
    cv::Mat lut = createLTU(NUM_CLASSES);
    cv::LUT(im_color, lut, im_color);
    cv::imwrite("color_map.png", im_color);
    // std::cout << "output color image is " << im_color << std::endl;
}

void hrnet_trt_destroy(void *h)
{
    HrnetTRTContext *trt_ctx;
 
    trt_ctx = (HrnetTRTContext *)h;
 
    // Release stream and buffers
    cudaStreamDestroy(trt_ctx->cuda_stream);
    CHECK(cudaFreeHost(trt_ctx->buffers[trt_ctx->inputIndex]));
    //cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]);
    CHECK(cudaFreeHost(trt_ctx->buffers[trt_ctx->outputIndex]));
    //cudaFree(trt_ctx->buffers[trt_ctx->outputIndex])
    // Destroy the engine
    trt_ctx->exe_context->destroy();
    trt_ctx->engine->destroy();
    trt_ctx->runtime->destroy();
 
    delete trt_ctx->data;
    delete trt_ctx->prob;
 
    delete trt_ctx;
}
