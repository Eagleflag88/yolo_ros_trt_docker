cmake_minimum_required(VERSION 2.8.11)

project(dl4ad)
# The version number.
set(DL4AD_MAJOR_VERSION 0)
set(DL4AD_MINOR_VERSION 0)
set(DL4AD_PATCH_VERSION 1)
set(DL4AD_VERSION
  ${DL4AD_MAJOR_VERSION}.${DL4AD_MINOR_VERSION}.${DL4AD_PATCH_VERSION})


find_package(CUDA REQUIRED)
find_package(OpenCV)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)


include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  "/usr/include/eigen3"
  )

# set up the output tree
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ===== yolo =====
aux_source_directory(${PROJECT_SOURCE_DIR}/src/yolo YOLO_SRC_DIR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yolov5_trt SHARED ${PROJECT_SOURCE_DIR}/include/yololayer.cu ${PROJECT_SOURCE_DIR}/src/yolo/yolov5_lib.cpp)
target_link_libraries(yolov5_trt 
                      nvinfer 
                      cudart) 

# ===== deepsort =====
aux_source_directory(${PROJECT_SOURCE_DIR}/src/deepsort DEEPSORT_SRC_DIR)
add_library(deepsort SHARED ${DEEPSORT_SRC_DIR})
target_link_libraries(deepsort
                      ${CUDA_LIBS} 
                      ${OpenCV_LIBS} 
                      cudart 
                      nvinfer 
                      nvonnxparser
)

# ===== hrnet =====
aux_source_directory(${PROJECT_SOURCE_DIR}/src/hrnet_seg/ HRNET_SRC_DIR)
add_library(hrnet_seg SHARED ${HRNET_SRC_DIR})
target_link_libraries(hrnet_seg
                      ${CUDA_LIBS} 
                      ${OpenCV_LIBS} 
                      cudart
                      nvinfer
                      nvonnxparser
)

# ===== ufld =====
aux_source_directory(${PROJECT_SOURCE_DIR}/src/ufld UFLD_SRC_DIR)
add_library(ufld SHARED ${UFLD_SRC_DIR})
target_link_libraries(ufld
                      ${CUDA_LIBS} 
                      ${OpenCV_LIBS} 
                      cudart
                      nvinfer
                      nvonnxparser
)

# ===== manager =====
aux_source_directory(${PROJECT_SOURCE_DIR}/src/manager M_SRC_DIR)
add_library(manager SHARED ${M_SRC_DIR})
target_link_libraries(manager
                      ${CUDA_LIBS} 
                      ${OpenCV_LIBS} 
                      cudart
                      nvinfer
                      nvonnxparser
                      )

# SET(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)

# target_include_directories(manager PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>)
# set_target_properties(mylib PROPERTIES PUBLIC_HEADER "include/mylib.h")
# install(TARGETS mylib
#     EXPORT mylib-targets
#     PUBLIC_HEADER DESTINATION include
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin)
# install(EXPORT mylib-targets
#     NAMESPACE mylib::
#     FILE mylib-config.cmake
#     DESTINATION lib/cmake/mylib)


