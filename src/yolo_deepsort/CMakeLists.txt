cmake_minimum_required(VERSION 3.0.2)
project(yolo_deepsort)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        image_transport
        nav_msgs
        roscpp
        rospy
        std_msgs
        tf
        message_generation
        )

find_package(CUDA REQUIRED)
find_package(OpenCV)

set(YOLO_PROJECT_DIR "/workspace/tensorrtx/yolov5")

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

catkin_package()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${YOLO_PROJECT_DIR}
        ${OpenCV_INCLUDE_DIRS}
)


cuda_add_executable(yolo_deepsort_node
        src/yolo_deepsort.cpp
        )
        
target_link_libraries(yolo_deepsort_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        nvinfer
        cudart
        myplugins
        )