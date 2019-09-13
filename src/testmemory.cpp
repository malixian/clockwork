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
#include <dmlc/logging.h>

using namespace clockwork::model;

void testmemory(int pagesize) {
}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

    int pagesize = 16 * 1024 * 1024;
	testmemory(pagesize);

	std::cout << "end" << std::endl;
}
