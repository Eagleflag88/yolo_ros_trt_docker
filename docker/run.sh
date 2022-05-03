#!/bin/bash
docker run \
    --gpus all \
    --net=host \
    -it --privileged --rm \
    -v /home/eagleflag/Documents/yolo_ros_trt_docker:/workspace/yolo_ros_trt_docker \
    yolov5_trt:latest

        # -v /home/eagleflag/Documents/MultiModalEvnPercption:/workspace/MultiModalEvnPercption \
            # --entrypoint /bin/bash \