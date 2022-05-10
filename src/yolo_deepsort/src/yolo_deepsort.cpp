/*
 * @Author: Eagleflag88 yijiang.xie@foxmail.com
 * @Date: 2022-05-06 18:24:23
 * @LastEditors: Eagleflag88 yijiang.xie@foxmail.com
 * @LastEditTime: 2022-05-07 11:17:43
 * @FilePath: /yolo_ros_trt_docker/src/yolo_deepsort/src/yolo_deepsort.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/image_encodings.h"

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
static ros::Subscriber compressed_image_sub;

//todo: 查看所有容器的使用，复制还是引用
//todo: lock for data race

char* yolo_engine = "/workspace/tensorrtx/yolov5/build/yolov5l.engine";
char* sort_engine = "/workspace/deepsort-tensorrt/resources/deepsort.engine";
char* hrnet_engine = "/workspace/tensorrtx/hrnet/hrnet-semantic-segmentation/build/hrnet_w48.engine";
char* ufld_engine = "/workspace/tensorrtx/ufld/build/lane_det.engine";
float conf_thre = 0.4;
Trtyolosort yosort(yolo_engine,
                   sort_engine, 
                   hrnet_engine,
                   ufld_engine);

static void CAM_Callback(const sensor_msgs::ImageConstPtr& img_msg_ptr);
static void Compressed_CAM_Callback(const sensor_msgs::CompressedImageConstPtr& img_msg_ptr);

int main(int argc, char **argv)
{

    ros::init(argc, argv, "cam_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    // Register the Subscriber
    // image_transport::Subscriber image_sub = it.subscribe("/kitti/camera_color_left/image_raw", 10, CAM_Callback);
    // image_transport::Subscriber image_sub = it.subscribe("cam_front/raw", 10, CAM_Callback);
    
    compressed_image_sub = nh.subscribe("/CAM_FRONT/image_rect_compressed", 10, Compressed_CAM_Callback);
    
    pub_image_track = nh.advertise<sensor_msgs::Image>("image_track", 1000);
    chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);

    map<int,vector<int>> personstate;
	map<int,int> classidmap;
	bool is_first = true;

    // cv::VideoCapture capture;
    // capture.open("/workspace/yolo_ros_trt_docker/Data/drving_through_toyky.mp4");  // 从视频文件读取
    // cv::Mat frame;  
    // std::vector<DetectBox> det;
    // cv::Mat frame_ld_out;
    // cv::Mat frame_ld_in;
    // cv::Mat frame_seg_in;
    // cv::Mat frame_seg_out;
    // while (capture.read(frame)) {
    //     frame_ld_in = frame.clone();
    //     frame_seg_in = frame.clone();
        
	// 	clock_t start_draw,end_draw;
    //     start_draw = clock();
    //     auto start = std::chrono::system_clock::now();
    //     yosort.TrtDetect(frame,conf_thre,det);
    //     auto end = std::chrono::system_clock::now();
    //     int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //     std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
    //     yosort.showDetection(frame, det);

    //     // Lane Detection
    //     yosort.TrtUfld(frame_ld_in, frame_ld_out);
    //     cv::waitKey(1);
    //     cv::imshow("ld_img", frame_ld_out);

    //     // // Semantic Segmentation
    //     // yosort.TrtSeg(frame_seg_in, frame_seg_out);
    //     // cv::waitKey(1);
    //     // cv::imshow("seg_img", frame_seg_out);
	// }
	
    //

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
    cv::Mat frame_seg_in = frame.clone();
    cv::Mat frame_ld_in = frame.clone();
    std::vector<DetectBox> det;
	auto start_draw_time = std::chrono::system_clock::now();

    clock_t start_draw,end_draw;
	start_draw = clock();
    auto start = std::chrono::system_clock::now();
    yosort.TrtDetect(frame,conf_thre,det);
    auto end = std::chrono::system_clock::now();
    int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
    yosort.showDetection(frame, det);
    
    // Semantic Segmentation
    cv::Mat frame_seg_out;
    yosort.TrtSeg(frame_seg_in, frame_seg_out);
    cv::waitKey(1);
    cv::imshow("seg_img", frame_seg_out);

    // Lane Detection
    cv::Mat frame_ld_out;
    yosort.TrtUfld(frame_ld_in, frame_ld_out);
    cv::waitKey(1);
    cv::imshow("ld_img", frame_ld_out);

}

static void Compressed_CAM_Callback(const sensor_msgs::CompressedImageConstPtr& img_msg_ptr)
{
    cv::Mat frame;
    try
    {
      cv_bridge::CvImagePtr cv_ptr_compressed = cv_bridge::toCvCopy(img_msg_ptr, sensor_msgs::image_encodings::BGR8);
      frame = cv_ptr_compressed->image;
    //   cv::imshow("imgCallback", imgCallback);
    //   cv::waitKey(1);
    //   cout<<"cv_ptr_compressed: "<<cv_ptr_compressed->image.cols<<" h: "<<cv_ptr_compressed->image.rows<<endl;
    }
    catch (cv_bridge::Exception& e)
    {
      //ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

    cv::Mat frame_seg_in = frame.clone();
    cv::Mat frame_ld_in = frame.clone();
    std::vector<DetectBox> det;
	auto start_draw_time = std::chrono::system_clock::now();

    clock_t start_draw,end_draw;
	start_draw = clock();
    auto start = std::chrono::system_clock::now();
    yosort.TrtDetect(frame,conf_thre,det);
    auto end = std::chrono::system_clock::now();
    int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
    yosort.showDetection(frame, det);
    
    // // Semantic Segmentation
    // cv::Mat frame_seg_out;
    // yosort.TrtSeg(frame_seg_in, frame_seg_out);
    // cv::waitKey(1);
    // cv::imshow("seg_img", frame_seg_out);

    // // Lane Detection
    // cv::Mat frame_ld_out;
    // yosort.TrtUfld(frame_ld_in, frame_ld_out);
    // cv::waitKey(1);
    // cv::imshow("ld_img", frame_ld_out);

}