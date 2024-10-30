# Use the nvidia official Ubuntu 22.04 as a base image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Add a non-root user
ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID
RUN <<EOF
    addgroup --gid ${REMOTE_GID} ${REMOTE_USER}
    adduser --disabled-password --uid ${REMOTE_UID} --gid ${REMOTE_GID} ${REMOTE_USER}
EOF
RUN apt-get update && apt-get install -y sudo \
    && echo "${REMOTE_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${REMOTE_USER} \
    && chmod 0440 /etc/sudoers.d/${REMOTE_USER}
ENV HOME /home/${REMOTE_USER}

# Install usufull packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y udev systemctl wget usbutils

# BEGIN freenect installation
# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libusb-1.0-0-dev \
    git \
    cmake

# Install optional packages
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    libxmu-dev \
    libxi-dev \
    libglib2.0-0

# Clone the Freecnet repository
RUN git clone https://github.com/OpenKinect/libfreenect /opt/freecnet

# Change working directory
RUN mkdir -p /opt/freecnet/build
WORKDIR /opt/freecnet/build

# Install the Python dependencies
RUN pip3 install numpy==2.1.2 cython

# Build and Install Freecnet
RUN cmake .. -DBUILD_PYTHON3=ON && make && make install
RUN cp /usr/local/lib/python3/dist-packages/freenect.so /usr/local/lib/python3.10/dist-packages/ \
    && ldconfig -v

# Install UDev rules
RUN cp /opt/freecnet/platform/linux/udev/51-kinect.rules /etc/udev/rules.d/
# END freenect installation

# BEGIN OpenCV installation
# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    yasm \
    libtbb2 \
    libtbb-dev \
    libpng-dev \
    libtiff-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libeigen3-dev \
    libgtk2.0-dev \
    pkg-config

ARG OPENCV_VERSION=4.9.0

RUN cd /opt/ && wget https://github.com/opencv/opencv/archive/4.9.0.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    mkdir /opt/opencv-${OPENCV_VERSION}/build

RUN cd /opt/opencv-${OPENCV_VERSION}/build && cmake \
    -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
    -DWITH_CUDNN=ON \
    -DWITH_CUBLAS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCUDA_FAST_MATH=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DWITH_TBB=ON \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_NVCUVID=OFF \
    -DWITH_NVCUVENC=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DWITH_CSTRIPES=ON \
    -DWITH_OPENCL=ON \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math" \
    # Install path will be /usr/local/lib (lib is implicit)
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    .. &&\
    make -j$(nproc)

RUN cd /opt/opencv-${OPENCV_VERSION}/build &&\
    make install &&\ 
    ldconfig
# END OpenCV installation

# Install Pre-commit and Git LFS
RUN pip install pre-commit black pytest
RUN apt update && apt install git-lfs -y

# Install other python dependencies
RUN pip3 install scipy matplotlib pandas scikit-learn numba torch
RUN pip3 install filterpy

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/workspace/src"

USER ${REMOTE_USER}
WORKDIR /workspace
