cmake_minimum_required(VERSION 3.4.1)

project(clockwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

include_directories(
	include
	$ENV{TVM_HOME}/include
	$ENV{TVM_HOME}/3rdparty/dmlc-core/include
	$ENV{TVM_HOME}/3rdparty/dlpack/include
	/usr/local/cuda/include
	${CMAKE_CURRENT_BINARY_DIR}
)

link_directories(
	$ENV{TVM_HOME}/build
	/usr/local/cuda/lib64
)

add_executable (main 
	src/main.cc
	src/clockwork/runtime.cpp
	src/clockwork/threadpoolruntime.cpp
	src/clockwork/greedyruntime.cpp
	src/clockwork/clockworkruntime.cpp
	src/clockwork/util.cpp
	src/clockwork/tvm_util.cpp
)

target_link_libraries(
	main 
	Threads::Threads 
	dl 
	cuda 
	cudart 
	tvm_runtime
	tbb
)
