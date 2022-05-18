#ifndef COMMON_HRNET_H_
#define COMMON_HRNET_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
//     DIR *p_dir = opendir(p_dir_name);
//     if (p_dir == nullptr) {
//         return -1;
//     }

//     struct dirent* p_file = nullptr;
//     while ((p_file = readdir(p_dir)) != nullptr) {
//         if (strcmp(p_file->d_name, ".") != 0 &&
//             strcmp(p_file->d_name, "..") != 0) {
//             //std::string cur_file_name(p_dir_name);
//             //cur_file_name += "/";
//             //cur_file_name += p_file->d_name;
//             std::string cur_file_name(p_file->d_name);
//             file_names.push_back(cur_file_name);
//         }
//     }

//     closedir(p_dir);
//     return 0;
// }


// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
void debug_print(ITensor *input_tensor, std::string head)
{
    std::cout << head << " : ";

    for (int i = 0; i < input_tensor->getDimensions().nbDims; i++)
    {
        std::cout << input_tensor->getDimensions().d[i] << " ";
    }
    std::cout << std::endl;
}
cv::Mat createLTU(int len)
{
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.data;
    for (int j = 0; j < 256; ++j)
    {
        p[j] = (j * (256 / len) > 255) ? uchar(255) : (uchar)(j * (256 / len));
    }
    return lookUpTable;
}
ITensor *MeanStd(INetworkDefinition *network, ITensor *input, float *mean, float *std, bool div255)
{
    if (div255)
    {
        Weights Div_225{DataType::kFLOAT, nullptr, 3};
        float *wgt = reinterpret_cast<float *>(malloc(sizeof(float) * 3));
        for (int i = 0; i < 3; ++i)
        {
            wgt[i] = 255.0f;
        }
        Div_225.values = wgt;
        IConstantLayer *d = network->addConstant(Dims3{3, 1, 1}, Div_225);
        input = network->addElementWise(*input, *d->getOutput(0), ElementWiseOperation::kDIV)->getOutput(0);
    }
    Weights Mean{DataType::kFLOAT, nullptr, 3};
    Mean.values = mean;
    IConstantLayer *m = network->addConstant(Dims3{3, 1, 1}, Mean);
    IElementWiseLayer *sub_mean = network->addElementWise(*input, *m->getOutput(0), ElementWiseOperation::kSUB);
    if (std != nullptr)
    {
        Weights Std{DataType::kFLOAT, nullptr, 3};
        Std.values = std;
        IConstantLayer *s = network->addConstant(Dims3{3, 1, 1}, Std);
        IElementWiseLayer *std_mean = network->addElementWise(*sub_mean->getOutput(0), *s->getOutput(0), ElementWiseOperation::kDIV);
        return std_mean->getOutput(0);
    }
    else
    {
        return sub_mean->getOutput(0);
    }
}

// 
#endif
