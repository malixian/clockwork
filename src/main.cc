#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "clockwork/runtime.h"
#include "clockwork/clockwork.h"
#include <sstream>
#include <atomic>
#include <thread>
#include <fstream>
#include <istream>
#include "clockwork/modeldata.h"
#include "clockwork/serializedmodel.h"
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>
#include "clockwork/tvm/decoupled_graph_runtime.h"
#include "clockwork/model.h"

using namespace clockwork;

void loadmodel() {

	const int dtype_code = kDLFloat;
	const int dtype_bits = 32;
	const int dtype_lanes = 1;
	const int device_type = kDLGPU;
	const int device_id = 0;

	std::string model = "/home/jcmace/modelzoo/resnet50/tesla-m40_batchsize1/tvm-model";

    ColdDiskModel* cold = new ColdDiskModel(
            model + ".so",
            model + ".clockwork",
            model + ".clockwork_params"
        );
    std::cout << "cold" << std::endl;

    CoolModel* cool = cold->load();
    std::cout << "cool" << std::endl;
    WarmModel* warm = cool->load();
    std::cout << "warm" << std::endl;

    void* ptr;
    CUDA_CALL(cudaMalloc(&ptr, warm->size()));
    std::cout << "warm malloc" << std::endl;

    HotModel* hot = warm->load(ptr);
    std::cout << "hot" << std::endl;

    hot->call();
    std::cout << "call" << std::endl;

}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

	loadmodel();

	std::cout << "end" << std::endl;
}
