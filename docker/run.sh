#!/bin/bash
###
 # @Author: Eagleflag88 yijiang.xie@foxmail.com
 # @Date: 2022-05-03 15:50:05
 # @LastEditors: Eagleflag88 yijiang.xie@foxmail.com
 # @LastEditTime: 2022-05-06 18:29:36
 # @FilePath: /docker/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

xhost +local:docker

docker run \
    --gpus all \
    --net=host \
    -it --privileged --rm \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /work/tools/yolo_ros_trt_docker:/workspace/yolo_ros_trt_docker \
    -v /work/tools/catkin_ws_dynamic_slam:/workspace/catkin_ws_dynamic_slam \
    -v /work/tools/catkin_ws_iceba:/workspace/catkin_ws_iceba \
    -v /work/tools/catkin_ws_svf:/workspace/catkin_ws_svf \
    -v /work/tools/catkin_ws_orbvins:/workspace/catkin_ws_orbvins \
    -v /work/tools/catkin_ws_hzp:/workspace/catkin_ws_hzp \
    -v /home/eagleflag/Downloads/semantic_mapping3d-0523/semantic_mapping:/workspace/semantic_mapping \
    -v /work/DataSet:/workspace/data \
    yolov5_trt:latest
    # yolov5_trt_py38:latest
    
    
        # eagleflag/yolo_tensorrt_ros:v1.0
        # -v /home/eagleflag/Documents/MultiModalEvnPercption:/workspace/MultiModalEvnPercption \
            # --entrypoint /bin/bash \

                # -v /work/tools/yolo_ros_trt_docker:/workspace/yolo_ros_trt_docker \