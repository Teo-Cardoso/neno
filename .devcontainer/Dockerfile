# Use the official Ubuntu 22.04 as a base image
FROM ubuntu:22.04

# Grant access to USB devices
RUN apt-get update && apt-get install -y udev systemctl

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libusb-1.0-0-dev \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install optional packages
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    libxmu-dev \
    libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the Freecnet repository
RUN git clone https://github.com/OpenKinect/libfreenect /opt/freecnet

# Change working directory
RUN mkdir -p /opt/freecnet/build
WORKDIR /opt/freecnet/build

# Install the Python dependencies
RUN pip3 install numpy cython

# Build and Install Freecnet
RUN cmake .. -DBUILD_PYTHON3=ON && make && make install
RUN cp /usr/local/lib/python3/dist-packages/freenect.so /usr/local/lib/python3.10/dist-packages/ \
    && ldconfig -v

# Install OpenCV for Python
ARG DEBIAN_FRONTEND=noninteractive
RUN pip3 install opencv-python==4.10.0.84 && \
    apt-get update && apt-get install libglib2.0-0 -y && rm -rf /var/lib/apt/lists/*

RUN umask 0000

# Install UDev rules
RUN cp /opt/freecnet/platform/linux/udev/51-kinect.rules /etc/udev/rules.d/

# Add folders to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/workspace/src"
RUN pip install pre-commit black pytest
RUN apt update && apt install git-lfs -y


WORKDIR /workspace
