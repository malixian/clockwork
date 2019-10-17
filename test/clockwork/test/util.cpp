#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <catch2/catch.hpp>
#include <nvml.h>
#include <iostream>

namespace clockwork{
namespace util {

std::string get_exe_location() {
    int bufsize = 1024;
    char buf[bufsize];
    int len = readlink("/proc/self/exe", buf, bufsize);
    return std::string(buf, len);
}

std::string get_clockwork_dir() {
    std::string exe_location = get_exe_location();
	return dirname(dirname(exe_location.data()));
}

std::string get_example_model(std::string name) {
    return get_clockwork_dir() + "/resources/" + name + "/model";
}

std::string get_example_batched_model(std::string name) {
    return get_example_model(name);
}

bool is_cuda_cache_disabled() {
    const char* v = std::getenv("CUDA_CACHE_DISABLE");
    if (v != nullptr) return std::string(v) == "1";
    return false;
}

bool is_force_ptx_jit_enabled() {
    const char* v = std::getenv("CUDA_FORCE_PTX_JIT");
    if (v != nullptr) return std::string(v) == "1";
    return false;
}

bool is_gpu_exclusive(int deviceId) {
    nvmlReturn_t status;

    status = nvmlInit();
    CHECK(status == NVML_SUCCESS);

    nvmlDevice_t device;
    status = nvmlDeviceGetHandleByIndex(deviceId, &device);
    CHECK(status == NVML_SUCCESS);

    nvmlComputeMode_t computeMode;
    status = nvmlDeviceGetComputeMode(device, &computeMode);
    CHECK(status == NVML_SUCCESS);

    status = nvmlShutdown();
    CHECK(status == NVML_SUCCESS);

    return computeMode == NVML_COMPUTEMODE_EXCLUSIVE_PROCESS;
}

std::pair<int, int> get_compute_capability(unsigned device_id) {
    nvmlReturn_t status;

    status = nvmlInit();
    CHECK(status == NVML_SUCCESS);

    nvmlDevice_t device;
    status = nvmlDeviceGetHandleByIndex(device_id, &device);
    CHECK(status == NVML_SUCCESS);

    int major, minor;
    status = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);

    status = nvmlShutdown();
    CHECK(status == NVML_SUCCESS);

    return std::make_pair(major, minor);
}

void nvml() {
    nvmlReturn_t status;

    status = nvmlInit();
    CHECK(status == NVML_SUCCESS);

    nvmlDevice_t device;
    status = nvmlDeviceGetHandleByIndex(0, &device);
    CHECK(status == NVML_SUCCESS);
    std::cout << " got device " << 0 << std::endl;

    nvmlComputeMode_t computeMode;
    status = nvmlDeviceGetComputeMode(device, &computeMode);
    CHECK(status == NVML_SUCCESS);
    std::cout << "compute mode is " << computeMode << std::endl;

    unsigned linkWidth;
    status = nvmlDeviceGetCurrPcieLinkWidth(device, &linkWidth);
    CHECK(status == NVML_SUCCESS);
    std::cout << "link width is " << linkWidth << std::endl;

    int major, minor;
    status = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
    std::cout << "Compute " << major << " " << minor << std::endl;

    status = nvmlShutdown();
    CHECK(status == NVML_SUCCESS);
}

}

namespace model {

std::vector<char*> make_cuda_pages(int page_size, int num_pages) {
    void* base_ptr;
    cudaError_t status = cudaMalloc(&base_ptr, page_size * num_pages);
    REQUIRE(status == cudaSuccess);
    std::vector<char*> pages(num_pages);
    for (unsigned i = 0; i < num_pages; i++) {
        pages[i] = static_cast<char*>(base_ptr) + i * page_size;
    }
    return pages;
}

void free_cuda_pages(std::vector<char*> pages) {
    cudaError_t status = cudaFree(pages[0]);
    REQUIRE(status == cudaSuccess);
}

void cuda_synchronize(cudaStream_t stream) {
	cudaError_t status = cudaStreamSynchronize(stream);
	REQUIRE(status == cudaSuccess);
}

}
}