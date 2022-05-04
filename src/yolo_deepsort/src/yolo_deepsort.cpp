//
// Created by eagleflag on 2021/3/21.
//

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <map>
#include "ros/ros.h"
#include "std_msgs/String.h"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "manager.hpp"

// Declaration of Publishers

static ros::Publisher pub_image_track;
static ros::Publisher chatter_pub;

// Declaration of Subscribers
static ros::Subscriber image_sub;

//todo: 查看所有容器的使用，复制还是引用
//todo: lock for data race

char* yolo_engine = "/workspace/tensorrtx/yolov5/build/yolov5l.engine";
char* sort_engine = "/workspace/deepsort-tensorrt/resources/deepsort.engine";
float conf_thre = 0.4;
Trtyolosort yosort(yolo_engine,sort_engine);

static void CAM_Callback(const sensor_msgs::ImageConstPtr& img_msg_ptr);

int main(int argc, char **argv)
{

    ros::init(argc, argv, "cam_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    // Register the Subscriber
    image_transport::Subscriber image_sub = it.subscribe("/kitti/camera_color_left/image_raw", 10, CAM_Callback);
    pub_image_track = nh.advertise<sensor_msgs::Image>("image_track", 1000);
    chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);

    map<int,vector<int>> personstate;
	map<int,int> classidmap;
	bool is_first = true;
	

    int count = 0;
    std_msgs::String msg;

    while(ros::ok())
    {
        std::stringstream status_msg;
        status_msg << "yolo_deepsort node working fine ";
        msg.data = status_msg.str();
        ROS_INFO("%s", msg.data.c_str());
        // chatter_pub.publish(msg);
        ros::spin();
        count++;
    }

    return 0;
}

static void CAM_Callback(const sensor_msgs::ImageConstPtr& img_msg_ptr)
{
    cv_bridge::CvImagePtr cam_cv_ptr = cv_bridge::toCvCopy(img_msg_ptr);
    ROS_INFO("Get a image from camera");
    // std::cout << "Number of column is " << cam_cv_ptr->image.cols << std::endl;
    cv::Mat frame;
    frame = cam_cv_ptr->image;

    std::vector<DetectBox> det;
	auto start_draw_time = std::chrono::system_clock::now();

    clock_t start_draw,end_draw;
	start_draw = clock();
    auto start = std::chrono::system_clock::now();
    yosort.TrtDetect(frame,conf_thre,det);
    auto end = std::chrono::system_clock::now();
    int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
    yosort.showDetection(frame,det);

}