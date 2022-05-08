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
include_directories(${PROJECT_SOURCE_DIR}/yolo/include)
include_directories(${PROJECT_SOURCE_DIR}/deepsort/include)
include_directories(${PROJECT_SOURCE_DIR}/hrnet_seg/include)
include_directories(${PROJECT_SOURCE_DIR}/ufld/include)
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3"
)

# ===== deepsort =====
aux_source_directory(${PROJECT_SOURCE_DIR}/deepsort/src SRC_DIR)
add_library(deepsort SHARED ${SRC_DIR})
target_link_libraries(deepsort 
    ${CUDA_LIBS} ${OpenCV_LIBS} 
    cudart nvinfer nvonnxparser
)

# ===== yolo =====
include_directories(${PROJECT_SOURCE_DIR}/yolo/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/yolo/src YOLO_SRC_DIR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yolov5_trt SHARED ${PROJECT_SOURCE_DIR}/yolo/include/yololayer.cu ${PROJECT_SOURCE_DIR}/yolo/src/yolov5_lib.cpp)
target_link_libraries(yolov5_trt nvinfer cudart deepsort) 

# ===== hrnet =====
aux_source_directory(${PROJECT_SOURCE_DIR}/hrnet_seg/src SRC_DIR)
add_library(hrnet_seg SHARED ${SRC_DIR})
target_link_libraries(hrnet_seg
                      ${CUDA_LIBS} 
                      ${OpenCV_LIBS} 
                      cudart
                      nvinfer
                      nvonnxparser
)

# ===== ufld =====
aux_source_directory(${PROJECT_SOURCE_DIR}/ufld/src SRC_DIR)
add_library(ufld SHARED ${SRC_DIR})
target_link_libraries(ufld
                      ${CUDA_LIBS} 
                      ${OpenCV_LIBS} 
                      cudart
                      nvinfer
                      nvonnxparser
)

# ===== main =====
aux_source_directory(${PROJECT_SOURCE_DIR}/src M_SRC_DIR)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(yolo_deepsort_node ${M_SRC_DIR})

target_link_libraries(yolo_deepsort_node 
                        nvinfer 
                        cudart 
                        yolov5_trt
                        hrnet_seg
                        ufld
                        ${catkin_LIBRARIES}
                        ${OpenCV_LIBS}
                        )