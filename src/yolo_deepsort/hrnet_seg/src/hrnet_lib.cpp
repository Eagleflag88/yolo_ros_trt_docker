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

static void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, int batchSize)
{
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
}

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
 
}HrnetTRTContext;

void * hrnet_trt_create(const char * engine_name)
{
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

    trt_ctx->data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    trt_ctx->prob = new float[BATCH_SIZE * OUTPUT_SIZE];
    trt_ctx->runtime = createInferRuntime(gLogger);
    assert(trt_ctx->runtime != nullptr);

    printf("hrnet_seg_trt_create  cuda engine... \n");
    trt_ctx->engine = trt_ctx->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(trt_ctx->engine != nullptr);
    trt_ctx->exe_context = trt_ctx->engine->createExecutionContext();

    delete[] trtModelStream;
    assert(trt_ctx->engine->getNbBindings() == 2);
 
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    trt_ctx->inputIndex = trt_ctx->engine->getBindingIndex(INPUT_BLOB_NAME);
    trt_ctx->outputIndex = trt_ctx->engine->getBindingIndex(OUTPUT_BLOB_NAME);
 
    assert(trt_ctx->inputIndex == 0);
    assert(trt_ctx->outputIndex == 1);

    // Create GPU buffers on device
    printf("hrnet_seg_trt_create  buffer ... \n");
    cudaSetDeviceFlags(cudaDeviceMapHost);
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
 
    printf("hrnet_seg_trt_create  stream ... \n");
    CUDA_CHECK(cudaStreamCreate(&trt_ctx->cuda_stream));
    //cudaStreamCreate(&trt_ctx->cuda_stream);
    printf("hrnet_seg_trt_create  done ... \n");
    return (void *)trt_ctx;
}

int hrnet_trt_seg(void *h, cv::Mat &img_BGR,cv::Mat& outimg)
{
    HrnetTRTContext *trt_ctx;
    int i;
    int delay_preprocess;
    int delay_infer;
 
    trt_ctx = (HrnetTRTContext *)h;
 
    auto start0 = std::chrono::system_clock::now();
 
    cv::Mat pr_img;
    cv::Mat img;
    cv::cvtColor(img_BGR, img, cv::COLOR_BGR2RGB);
    cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H));
    img = pr_img.clone(); // for img show
    pr_img.convertTo(pr_img, CV_32FC3);
    if (!pr_img.isContinuous())
    {
        pr_img = pr_img.clone();
    }
    std::memcpy(trt_ctx->data, pr_img.data, BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float));

    cudaHostGetDevicePointer(&trt_ctx->buffers[trt_ctx->inputIndex], trt_ctx->data, 0);  // buffers[inputIndex]-->data
    cudaHostGetDevicePointer(&trt_ctx->buffers[trt_ctx->outputIndex], trt_ctx->prob, 0); // buffers[outputIndex] --> prob
    
    doInference(*trt_ctx->exe_context, trt_ctx->cuda_stream, trt_ctx->buffers, BATCH_SIZE);

    outimg = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);
    for (int row = 0; row < INPUT_H; ++row)
    {
        uchar *uc_pixel = outimg.data + row * outimg.step;
        for (int col = 0; col < INPUT_W; ++col)
        {
            uc_pixel[col] = (uchar)trt_ctx->prob[row * INPUT_W + col];
        }
    }
}

void hrnet_trt_destroy(void *h)
{
    HrnetTRTContext *trt_ctx;
 
    trt_ctx = (HrnetTRTContext *)h;
 
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
