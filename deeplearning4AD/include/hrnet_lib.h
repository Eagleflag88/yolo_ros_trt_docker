
#pragma once 
#include <opencv2/opencv.hpp>
 
void * hrnet_trt_create(const char * engine_name);
 
int hrnet_trt_seg(void *h, cv::Mat &img_BGR,cv::Mat& outimg);
 
void hrnet_trt_destroy(void *h);
 



