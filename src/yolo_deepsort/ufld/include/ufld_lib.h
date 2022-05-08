
#pragma once 
#include <opencv2/opencv.hpp>
 
void * ufld_trt_create(const char * engine_name);
 
int ufld_trt_det(void *h, cv::Mat &img,cv::Mat& outimg);
 
void ufld_trt_destroy(void *h);
 



