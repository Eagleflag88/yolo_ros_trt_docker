FROM nvcr.io/nvidia/tensorrt:20.03-py3

RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get clean
RUN apt-get -y update --fix-missing

# git clone tensorrtx
RUN git clone https://github.com/wang-xinyu/tensorrtx.git

# install pytorch
RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py
RUN python3 get-pip.py --force-reinstall
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install pip -U
RUN pip3 install torch torchvision torchaudio 

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

RUN apt install software-properties-common -y
RUN add-apt-repository -y ppa:timsc/opencv-3.4
RUN apt-get update
RUN apt-get install -y libopencv-dev libopencv-dnn-dev libopencv-shape3.4-dbg 

# Build tensorrtx/yolov5 and run
RUN mkdir /usr/include/opencv
RUN cd /workspace/tensorrtx/yolov5/ && \
    mkdir build && \
    cd build && \
    cp /workspace/yolov5/yolov5s.wts /workspace/tensorrtx/yolov5/build/ && \
    pwd && \
    cmake .. && \
    make

# Install ROS-Melodic
# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO melodic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-desktop \
    && rm -rf /var/lib/apt/lists/*

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

RUN apt update -y && apt upgrade 
RUN apt install -y mlocate ros-melodic-pcl-conversions ros-melodic-pcl-ros

# RUN git clone https://github.com/Eagleflag88/yolo_ros_trt_docker.git
# RUN . /opt/ros/melodic/setup.sh && \
#     cd /workspace/yolo_ros_trt_docker && \
#     catkin_make -j2

# CMD ["/bin/bash", "-c", "./tensorrtx/yolov5/build/yolov5 -s tensorrtx/yolov5/build/yolov5s.wts tensorrtx/yolov5/build/yolov5s.engine s"]

    



    



    
