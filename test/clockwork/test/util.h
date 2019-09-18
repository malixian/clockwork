#ifndef _CLOCKWORK_TEST_UTIL_H_
#define _CLOCKWORK_TEST_UTIL_H_

#include <unistd.h>
#include <libgen.h>
#include <string>
#include <cuda_runtime.h>
#include <catch2/catch.hpp>
#include "clockwork/util.h"

namespace clockwork{
namespace util {

std::string get_exe_location() {
    int bufsize = 1024;
    char buf[bufsize];
    int len = readlink("/proc/self/exe", buf, bufsize);
    return std::string(buf, len);
}

std::string get_clockwork_dir() {
	return dirname(dirname(get_exe_location().data()));
}

std::string get_example_model() {
	return get_clockwork_dir() + "/resources/resnet18_tesla-m40_batchsize1/model";
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

void cuda_synchronize() {
	cudaError_t status = cudaStreamSynchronize(clockwork::util::Stream());
	REQUIRE(status == cudaSuccess);
}

}
}

#endif