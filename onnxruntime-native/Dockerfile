FROM debian:bookworm-slim as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        unzip \
        ca-certificates \
        bash \
        libssl-dev \
        libffi-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        liblzma-dev && \
    rm -rf /var/lib/apt/lists/*

# ----- Python Layer -----
FROM base as python
ARG PYTHON_VERSION=3.10.13

RUN curl -LO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

# ----- CMake Layer -----
FROM python as cmake
ARG CMAKE_VERSION=3.28.3

RUN curl -LO https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar --strip-components=1 -xz -C /usr/local -f cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

# ----- Build ONNX Runtime -----
FROM cmake as builder
WORKDIR /workspace

# Clone separately for caching
RUN git clone --branch v1.22.0 --depth 1 https://github.com/microsoft/onnxruntime onnxruntime-src

RUN mkdir build && \
    cd build && \
    cmake ../onnxruntime-src/cmake \
      -Donnxruntime_ENABLE_PYTHON=OFF \
      -Donnxruntime_BUILD_SHARED_LIB=ON \
      -Donnxruntime_BUILD_UNIT_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -Donnxruntime_ENABLE_LTO=ON \
      -Donnxruntime_ENABLE_CPU=ON \
      -Donnxruntime_ENABLE_CUDA=OFF \
      -Donnxruntime_ENABLE_DML=OFF && \
    make -j 8

# ----- Final Export Stage -----
FROM scratch as export

COPY --from=builder /workspace/build/Release/libonnxruntime.so /libonnxruntime.so
