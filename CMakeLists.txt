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

file(GLOB MANAGED_RUNTIME_SRCS src/clockwork/multitenant/*.cc)
add_library(multitenant ${MANAGED_RUNTIME_SRCS})

add_executable (main
	src/main.cc
	src/clockwork/clockwork_queue.cpp )

target_link_libraries(
	main
	multitenant
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
)
