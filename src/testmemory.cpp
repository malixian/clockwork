#include <iostream>
#include <sstream>
#include <atomic>
#include <thread>
#include <fstream>
#include <istream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include "clockwork/util/util.h"
#include "clockwork/util/tvm_util.h"
#include <tvm/runtime/cuda_common.h>
#include "clockwork/model/memory.h"

using namespace clockwork::model;

void testmemory() {
    size_t total_size = 100;
    size_t page_size = 11;
    void* baseptr = malloc(total_size);

    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size);

}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

	testmemory();

	std::cout << "end" << std::endl;
}
