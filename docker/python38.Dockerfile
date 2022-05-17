FROM nvcr.io/nvidia/tensorrt:21.05-py3

RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get clean
RUN apt-get -y update --fix-missing

# install pytorch
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install pip -U
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# git clone tensorrtx
RUN git clone https://github.com/Eagleflag88/tensorrtx.git

# Download yolov5
RUN git clone -b v6.0 https://github.com/ultralytics/yolov5.git
RUN pip3 install pandas requests opencv-python pyyaml tqdm matplotlib seaborn
RUN cp /workspace/tensorrtx/yolov5/gen_wts.py /workspace/yolov5/
RUN cd yolov5/ && \
    ls && \
    wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
RUN apt upgrade -y && apt install libgl1-mesa-glx -y
RUN cd yolov5/ && \
    python3 gen_wts.py -w yolov5s.pt -o yolov5s.wts

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y software-properties-common

RUN git clone -b 3.4 https://github.com/opencv/opencv
RUN git clone -b 3.4 https://github.com/opencv/opencv_contrib.git
RUN cd /workspace/opencv/ && \
    mkdir build && \
    cd build && \
    cmake -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules .. && \
    make -j4 && \
    make install 

# Build tensorrtx/yolov5 and run
RUN mkdir /usr/include/opencv
RUN cd /workspace/tensorrtx/yolov5/ && \
    mkdir build && \
    cd build && \
    cp /workspace/yolov5/yolov5s.wts /workspace/tensorrtx/yolov5/build/ && \
    pwd && \
    cmake .. && \
    make

# Install ROS-Noetic

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# # setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop \
    && rm -rf /var/lib/apt/lists/*

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

RUN apt update -y && apt upgrade 
RUN apt install -y mlocate ros-noetic-pcl-conversions ros-noetic-pcl-ros

# # Install detr
# RUN git clone https://github.com/facebookresearch/detr.git
# RUN cd /workspace/detr && \
#     wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
    
# RUN cp /workspace/tensorrtx/detr/gen_wts.py /workspace/detr/ && \
#     pip3 install packaging 
# RUN cd /workspace/detr && \
#     python3 gen_wts.py

RUN ./tensorrtx/yolov5/build/yolov5 -s tensorrtx/yolov5/build/yolov5s.wts tensorrtx/yolov5/build/yolov5s.engine s


# Install deep sort pytorch
RUN git clone https://github.com/Eagleflag88/deep_sort_pytorch.git
RUN git clone https://github.com/Eagleflag88/deepsort-tensorrt.git

RUN cd deep_sort_pytorch/ && \
    git submodule update --init --recursive

RUN pip3 install easydict

RUN cp /workspace/deepsort-tensorrt/exportOnnx.py /workspace/deep_sort_pytorch/ && \
    cd /workspace/deep_sort_pytorch/ && \
    python3 exportOnnx.py && \
    mv deepsort.onnx /workspace/deepsort-tensorrt/resources

RUN cd /workspace/deepsort-tensorrt/ && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    ./onnx2engine ../resources/deepsort.onnx ../resources/deepsort.engine

RUN git clone https://github.com/Eagleflag88/yolov5-deepsort-tensorrt.git

RUN cd /workspace/yolov5-deepsort-tensorrt/ && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make 


# Install HRNet
RUN git clone https://github.com/Eagleflag88/HRNet-Semantic-Segmentation.git

RUN cd /workspace/HRNet-Semantic-Segmentation/model && \
    cat hrnet0* > hrnet.tar.gz && \
    tar xvzf hrnet.tar.gz && \
    cp /workspace/tensorrtx/hrnet/hrnet-semantic-segmentation/gen_wts.py /workspace/HRNet-Semantic-Segmentation/tools/ && \
    cd /workspace/HRNet-Semantic-Segmentation/ && \
    pip3 install yacs && \
    python3 tools/gen_wts.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml --ckpt_path model/hrnet/hrnet_w48_cityscapes_cls19_1024x2048_trainset_pytorch_v11.pth --save_path hrnet_w48.wts && \
    cp hrnet_w48.wts /workspace/tensorrtx/hrnet/hrnet-semantic-segmentation/

RUN cd /workspace/tensorrtx/hrnet/hrnet-semantic-segmentation/ && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    ./hrnet -s ../hrnet_w48.wts hrnet_w48.engine 48 && \
    ./hrnet -d  ./hrnet_w48.engine ../../../ufld/samples/

# # Install swin transformer -> Unsuccessful
# RUN git clone https://github.com/Eagleflag88/Swin-Transformer.git

# RUN cd /workspace/Swin-Transformer/models && \
#     cat swin_transformer0* > swin_transformer.tar.gz && \
#     tar xvzf swin_transformer.tar.gz && \
#     cp /workspace/tensorrtx/swin-transformer/semantic-segmentation/gen_wts.py /workspace/Swin-Transformer/

# Install ulfd

RUN apt install -y iputils-ping

RUN git clone https://github.com/Eagleflag88/Ultra-Fast-Lane-Detection.git


RUN cp /workspace/tensorrtx/ufld/gen_wts.py /workspace/Ultra-Fast-Lane-Detection/ && \
    cd /workspace/Ultra-Fast-Lane-Detection/model && \
    cat tusimple0* > tusimple.tar.gz && \
    tar xvzf tusimple.tar.gz && \
    mv tusimple/tusimple_18.pth /workspace/Ultra-Fast-Lane-Detection/ && \
    cd /workspace/Ultra-Fast-Lane-Detection/ && \
    python3 gen_wts.py && \
    mv lane.wts /workspace/tensorrtx/ufld/

RUN cd /workspace/tensorrtx/ufld/ && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    ./lane_det -s && \
    ./lane_det -d ../samples/

RUN cd yolov5/ && \
    ls && \
    wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l.pt

RUN cd yolov5/ && \
    python3 gen_wts.py -w yolov5l.pt -o yolov5l.wts && \
    cp yolov5l.wts /workspace/tensorrtx/yolov5/build/

RUN ./tensorrtx/yolov5/build/yolov5 -s tensorrtx/yolov5/build/yolov5l.wts tensorrtx/yolov5/build/yolov5l.engine l


# RUN git clone https://github.com/Eagleflag88/yolo_ros_trt_docker.git
# RUN . /opt/ros/noetic/setup.sh && \
#     cd /workspace/yolo_ros_trt_docker && \
#     catkin_make -j2









    



    
