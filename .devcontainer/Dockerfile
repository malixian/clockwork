# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.145.1/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version: bionic, focal
# ARG VARIANT="focal"
# FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}
FROM nvidia/cuda:11.1-base

RUN DEBIAN_FRONTEND="noninteractive" apt-get update -yqq && apt-get upgrade -yqq
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --fix-missing -yqq git cmake gcc-8 g++-8 g++ libtbb-dev libasio-dev libconfig++ libboost-filesystem-dev autoconf automake libtool curl make unzip libboost-all-dev clang llvm sudo nvidia-cuda-dev

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

ARG ncpus

WORKDIR /
RUN git clone --recursive -b v3.12.0 https://github.com/protocolbuffers/protobuf.git
# make protobuf
WORKDIR /protobuf
RUN ./autogen.sh
RUN ./configure
RUN make -j ${ncpus}
RUN make install
RUN ldconfig

WORKDIR /
RUN git clone --depth=1 --recursive --single-branch --branch clockwork-v0.6 https://github.com/0xreza/tvm.git
WORKDIR /tvm/build
RUN sed -i 's/set(USE_CUDA ON)/set(USE_CUDA OFF)/' config.cmake
RUN cmake ..
RUN make -j ${ncpus}
RUN make install

ENV TVM_HOME=/tvm
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TVM_HOME/build
ENV DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$TVM_HOME/build
