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
#include "clockwork/pagedmodeldef.h"
#include <dmlc/logging.h>

using namespace clockwork::model;

void testmemory(int pagesize) {

    std::string model = "/home/jcmace/modelzoo/resnet50/tesla-m40_batchsize1/tvm-model";
    std::string data;
    clockwork::util::readFileAsString(model+".clockwork", data);

    std::string weights;
    clockwork::util::readFileAsString(model+".clockwork_params", weights);

    ModelDef modeldef;
    ModelDef::ReadFrom(data, modeldef);


    CHECK(weights.size() == modeldef.weights_memory) << "Inconsistent weights size " << weights.size() << " " << modeldef.weights_memory;
    processModelDef(modeldef, pagesize, weights.c_str());
}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

    int pagesize = 16 * 1024 * 1024;
	testmemory(pagesize);

	std::cout << "end" << std::endl;
}
