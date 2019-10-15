cmake_minimum_required(VERSION 3.4.1)

project(clockwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

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
	${Boost_INCLUDE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
)

link_directories(
	$ENV{TVM_HOME}/build
	$ENV{TVM_HOME}/3rdparty/dmlc-core/build
	/usr/local/cuda/lib64
)

add_library(clockwork
	src/clockwork/task.cpp
	src/clockwork/modeldef.cpp
	src/clockwork/common.cpp
	src/clockwork/cache.cpp
	src/clockwork/util.cpp
	src/clockwork/action.cpp
	src/clockwork/runtime.cpp
	src/clockwork/memory.cpp
	src/clockwork/worker.cpp
	src/clockwork/model/memfile.cpp
	src/clockwork/model/model.cpp
	src/clockwork/model/cuda.cpp
	src/clockwork/model/so.cpp
	src/clockwork/api/worker_api.cpp
	src/clockwork/network/network.cpp
	src/clockwork/network/client_api.cpp
	src/clockwork/network/worker_net.cpp
	src/clockwork/network/client_net.cpp
	src/clockwork/alternatives/runtime_model.cpp
	src/clockwork/alternatives/model_manager.cpp
	src/clockwork/alternatives/threadpoolruntime.cpp
	src/clockwork/alternatives/greedyruntime.cpp
	src/clockwork/alternatives/worker.cpp
)

target_link_libraries(clockwork PUBLIC
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
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
	nvidia-ml
)

add_executable (cudafatbin
	src/cudafatbin.cc
	src/clockwork/util.cpp
)

target_link_libraries(
	cudafatbin
	clockwork
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
)

include_directories(clockwork-convert)

add_executable (convert
	src/clockwork/tvm/decoupled_graph_runtime.cc
	src/clockwork-convert/tvm_model.cpp
	src/clockwork-convert/tvm_abstract_model.cpp
	src/clockwork-convert/convert.cpp
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
	nvidia-ml
)

add_executable (inflate
	src/inflate.cpp
)

target_link_libraries(
	inflate
	clockwork
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
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
	nvidia-ml
)

include_directories(test)

add_executable (tests
	test/tests.cpp
	test/clockwork/test/util.cpp
	test/clockwork/test/actions.cpp
	test/clockwork/test/testtask.cpp
	test/clockwork/test/testaction.cpp
	test/clockwork/test/testworker.cpp
	test/clockwork/test/testcache.cpp
	test/clockwork/test/testpriorityqueue.cpp
	test/clockwork/test/testclient.cpp
	test/clockwork/test/model/testmodel.cpp
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
	nvidia-ml
)

add_executable (profile
	profile/profile.cpp
	test/clockwork/test/util.cpp
	profile/clockwork/profile/check.cpp
	profile/clockwork/profile/alternatives/profilegreedy.cpp
	profile/clockwork/profile/model/profilecuda.cpp
	profile/clockwork/profile/model/profilemodel.cpp
)

target_link_libraries(
	profile
	clockwork
	Threads::Threads
	dl
	cuda
	cudart
	tvm_runtime
	tbb
	nvidia-ml
)

# protobuf imports
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/clockwork.proto)
add_library(proto ${PROTO_HDRS} ${PROTO_SRCS})
target_link_libraries(proto INTERFACE ${Protobuf_LIBRARIES})


add_executable (worker
	src/worker.cpp
)

target_link_libraries(
    worker
    clockwork
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    proto
    ${Boost_SYSTEM_LIBRARY}
)


add_executable (controller
	src/controller.cpp
)

target_link_libraries(
    controller
    clockwork
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    proto
    ${Boost_SYSTEM_LIBRARY}
)


add_executable (client
	src/client.cpp
)

target_link_libraries(
    client
    clockwork
    Threads::Threads
    dl
    cuda
    cudart
    tvm_runtime
    proto
    ${Boost_SYSTEM_LIBRARY}
)

