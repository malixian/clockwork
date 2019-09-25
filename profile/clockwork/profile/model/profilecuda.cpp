#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/so.h"
#include "clockwork/model/cuda.h"

using namespace clockwork;


void check_environment() {
    bool environmentIsOK = true;
    if (!util::is_cuda_cache_disabled()) {
        std::cout << "✘ CUDA cache is enabled!  It should be disabled by setting environment variable CUDA_CACHE_DISABLE=1" << std::endl;
        environmentIsOK = false;
    } else {
        std::cout << "✔ CUDA cache is disabled" << std::endl;
    }
    if (util::is_force_ptx_jit_enabled()) {
        std::cout << "✘ PTX JIT is being forced!  Unset the CUDA_FORCE_PTX_JIT environment variable" << std::endl;
        environmentIsOK = false;
    } else {
        std::cout << "✔ PTX JIT is not forced" << std::endl;
    }
    REQUIRE(environmentIsOK);
}

TEST_CASE("Check environment variables", "[profile] [check]") {
    check_environment();
}

void fill_memory(size_t &total_malloced, size_t &peak_usage) {
    size_t cudaMallocSize = 16 * 1024 * 1024;
    total_malloced = 0;

    cudaError_t status;
    
    size_t free, total;
    status = cudaMemGetInfo(&free, &total);
    REQUIRE((status == cudaSuccess));
    size_t initialUsed = total-free;

    std::vector<void*> ptrs;
    for (unsigned i = 0; true; i++) {
        void* ptr;
        status = cudaMalloc(&ptr, cudaMallocSize);
        REQUIRE((status == cudaSuccess || status == cudaErrorMemoryAllocation));

        if (status == cudaErrorMemoryAllocation) {
            status = cudaMemGetInfo(&free, &total);
            REQUIRE((status == cudaSuccess));
            peak_usage = total - free;

            break;
        } else {
            ptrs.push_back(ptr);
            total_malloced += cudaMallocSize;
        }
    }
    for (void* &ptr : ptrs) {
        status = cudaFree(ptr);
        REQUIRE(status == cudaSuccess);
    }

    status = cudaMemGetInfo(&free, &total);
    REQUIRE((status == cudaSuccess));
    REQUIRE( (total-free) == initialUsed );
}

TEST_CASE("Profile memory limit for cudaMalloc", "[profile] [cudaMalloc]") {
    util::initializeCudaStream(0);
    size_t total_malloced = 0;
    size_t peak_usage = 0;
    fill_memory(total_malloced, peak_usage);
    std::cout << "cudaMalloc total=" << total_malloced << " plus " << (peak_usage - total_malloced) << " additional" << std::endl;

    void* ptr;
    cudaError_t status;
    status = cudaMalloc(&ptr, total_malloced);
    REQUIRE(status == cudaSuccess);
    status = cudaFree(ptr);
    REQUIRE(status == cudaSuccess);
}

void profile_model(std::string model_name, std::string model_path, int expected_blob_size) {
    std::string so_filename = model_path + ".so";
    so::SharedObject so(so_filename);
    
    const char* cuda_blob = reinterpret_cast<const char*>(so.GetSymbol(tvm::runtime::symbol::tvm_dev_mblob));
    REQUIRE(cuda_blob != nullptr);

    cudaError_t status;
    util::initializeCudaStream(0);

    size_t free, total;
    status = cudaMemGetInfo(&free, &total);
    REQUIRE(status == cudaSuccess);
    size_t initial_use = total-free;

    std::cout << "Profiling " << model_name << " with initial memory used=" << (total-free) << std::endl;

    std::vector<cuda::UnloadedCUDAModule*> unloaded;
    std::vector<cuda::LoadedCUDAModule*> loaded;

    int maxIterations = 1000000;
    for (unsigned i = 0; i < maxIterations; i++) {
        cuda::UnloadedCUDAModule* unloaded_cuda = new cuda::UnloadedCUDAModule(cuda_blob);
        REQUIRE(unloaded_cuda->data.size() == expected_blob_size);

        CUmodule module;
        CUresult result = cuModuleLoadFatBinary(&module, unloaded_cuda->data.c_str());
        REQUIRE((result == CUDA_SUCCESS || result == CUDA_ERROR_DEINITIALIZED || result == CUDA_ERROR_OUT_OF_MEMORY));

        if (result == CUDA_ERROR_OUT_OF_MEMORY) {
            size_t free, total;
            status = cudaMemGetInfo(&free, &total);
            REQUIRE(status == cudaSuccess);
            std::cout << model_name << ": limit at n=" << i << ", size=" << unloaded_cuda->data.size() << ", total used=" << (total-free) << ", average=" << ((total-free)/i) << std::endl;
            break;
        }

        cuda::LoadedCUDAModule* loaded_cuda = new cuda::LoadedCUDAModule(unloaded_cuda, module);

        unloaded.push_back(unloaded_cuda);
        loaded.push_back(loaded_cuda);

        if (i % 1000 == 0) {
            status = cudaMemGetInfo(&free, &total);
            REQUIRE(status == cudaSuccess);
            std::cout << " ... " << model_name << " iteration " << i << " memory used is " << (total-free) << std::endl;
        }
    }

    // See how much more memory we can malloc
    size_t total_malloced = 0;
    size_t peak_usage = 0;
    fill_memory(total_malloced, peak_usage);
    std::cout << "cudaMalloc additional total=" << total_malloced << " for peak usage of " << peak_usage << std::endl;;

    for (auto &l : loaded) {
        l->unload();
    }
    for (auto &u : unloaded) {
        delete u;
    }

    status = cudaMemGetInfo(&free, &total);
    REQUIRE(status == cudaSuccess);
    REQUIRE( (total-free) == initial_use );
}

TEST_CASE("Profile memory limit for cuModuleLoad - resnet18", "[profile] [resnet18] [cuModuleLoad]") {
    std::string model_name = "resnet18";
    std::string model_path = clockwork::util::get_example_model("resnet18_tesla-m40_batchsize1");
    int expected_blob_size = 388408;
    profile_model(model_name, model_path, expected_blob_size);
}

TEST_CASE("Profile memory limit for cuModuleLoad - resnet50", "[profile] [resnet50] [cuModuleLoad]") {
    std::string model_name = "resnet50";
    std::string model_path = clockwork::util::get_example_model("resnet50_tesla-m40_batchsize1");
    int expected_blob_size = 403360;
    profile_model(model_name, model_path, expected_blob_size);
}