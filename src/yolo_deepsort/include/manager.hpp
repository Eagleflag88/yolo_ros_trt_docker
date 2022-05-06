/*
 * @Author: Eagleflag88 yijiang.xie@foxmail.com
 * @Date: 2022-05-06 18:24:23
 * @LastEditors: Eagleflag88 yijiang.xie@foxmail.com
 * @LastEditTime: 2022-05-06 19:36:56
 * @FilePath: /yolo_ros_trt_docker/src/yolo_deepsort/include/manager.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef _MANAGER_H
#define _MANAGER_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "deepsort.h"
#include "logging.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "time.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "yolov5_lib.h"
#include "deepsort.h"
#include "hrnet_lib.h"

using std::vector;
using namespace cv;
//static Logger gLogger;

class Trtyolosort{
public:
	// init 
	Trtyolosort(char *yolo_engine_path,char *sort_engine_path, char *hrnet_engine_path);
	// detect and show
	int TrtDetect(cv::Mat &frame,float &conf_thresh,std::vector<DetectBox> &det);
	void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes);
	
	// segment and show
	void TrtSeg(cv::Mat &frame_in, cv::Mat &frame_out);

private:
	char* yolo_engine_path_ = NULL;
	char* sort_engine_path_ = NULL;
    void *trt_engine = NULL;
	char* hrnet_engine_path_ = NULL;
	void *hrnet_trt_engine = NULL;
    // deepsort parms
    DeepSort* DS;
    std::vector<DetectBox> t;
};
#endif  // _MANAGER_H

