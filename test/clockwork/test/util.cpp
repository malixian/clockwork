#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <catch2/catch.hpp>

namespace clockwork{
namespace util {

std::string get_exe_location() {
    int bufsize = 1024;
    char buf[bufsize];
    int len = readlink("/proc/self/exe", buf, bufsize);
    return std::string(buf, len);
}

std::string get_clockwork_dir() {
    int bufsize = 1024;
    char buf[bufsize];
    int len = readlink("/proc/self/exe", buf, bufsize);
	return dirname(dirname(buf));
}

std::string get_example_model(std::string name) {
    return get_clockwork_dir() + "/resources/" + name + "/model";
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