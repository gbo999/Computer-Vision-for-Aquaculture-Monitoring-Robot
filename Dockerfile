# Use the official CUDA base image

# Increase Timeout and Retries for apt-get






FROM nvidia/cuda:12.3.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends tzdata git && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN git config --global http.postBuffer 524288000  # Set buffer to 500MB

RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "60";' > /etc/apt/apt.conf.d/75timeout
# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-numpy \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libeigen3-dev \
    libhdf5-dev \
    doxygen

# Clone OpenCV repository
RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout 4.9.0  # Replace <opencv_version> with the desired version

# Clone OpenCV contrib repository
RUN git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv_contrib && \
    git checkout 4.9.0  # Replace <opencv_version> with the desired version

# Build OpenCV
RUN cd opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_CUDA=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          .. && \
    make -j$(nproc) && \
    make install

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Cleanup
RUN rm -rf opencv opencv_contrib

# Set the working directory
