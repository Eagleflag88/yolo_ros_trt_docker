#!/bin/bash
###
 # @Author: Eagleflag88 yijiang.xie@foxmail.com
 # @Date: 2022-05-03 15:50:05
 # @LastEditors: Eagleflag88 yijiang.xie@foxmail.com
 # @LastEditTime: 2022-05-04 01:18:57
 # @FilePath: /yolo_ros_trt_docker/docker/build.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# docker build --tag yolov5_trt -f Dockerfile .
docker build --tag yolov5_trt_py38 -f python38.Dockerfile .

