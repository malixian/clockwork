cmake_minimum_required(VERSION 3.4.1)

project(clockwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(Threads REQUIRED)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

include_directories(
	include
	src
	external
	$ENV{TVM_HOME}/include
	$ENV{TVM_HOME}/3rdparty/dmlc-core/include
	$ENV{TVM_HOME}/3rdparty/dlpack/include
	/usr/local/cuda/include
	${CMAKE_CURRENT_BINARY_DIR}
)

link_directories(
	$ENV{TVM_HOME}/build
	$ENV{TVM_HOME}/3rdparty/dmlc-core/build
	/usr/local/cuda/lib64
)

add_library(clockwork
	src/clockwork/runtime.cpp
	src/clockwork/clockworkruntime.cpp
	src/clockwork/cache.cpp
	src/clockwork/runtime_model.cpp
	src/clockwork/util.cpp
	src/clockwork/model/memfile.cpp
	src/clockwork/model/model_impl.cpp
	src/clockwork/model/cuda.cpp
	src/clockwork/model/so.cpp
	src/clockwork/alternatives/threadpoolruntime.cpp
	src/clockwork/alternatives/greedyruntime.cpp
	src/clockwork/alternatives/worker.cpp
)

target_link_libraries(clockwork PUBLIC
	cuda
	cudart
	tvm_runtime
	tbb
)

add_executable (main
	src/main.cc
	src/clockwork/util.cpp
)

target_link_libraries(
	main
	clockwork
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
)

add_executable (convert
	src/clockwork/tvm/decoupled_graph_runtime.cc
	src/clockwork-convert/convert.cpp
	src/clockwork-convert/convert_to_paged.h
)

target_link_libraries(
	convert
	clockwork
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
)

add_executable (testmemory
	src/testmemory.cpp
)

target_link_libraries(
	testmemory
	clockwork
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
)

add_executable (tests
	test/tests.cpp
	test/clockwork/testmemory.cpp
)

target_link_libraries(
	tests
	clockwork
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
)
