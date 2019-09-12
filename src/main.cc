#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "clockwork/runtime.h"
#include "clockwork/clockwork.h"
#include <sstream>
#include <atomic>
#include <thread>
#include <fstream>
#include <istream>
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
#include <cuda_runtime.h>
#include <chrono>
#include "clockwork/util/util.h"
#include "clockwork/util/tvm_util.h"
#include <tvm/runtime/cuda_common.h>


struct ProfileData {
    float params, input, exec, output;
    std::chrono::high_resolution_clock::time_point cool, warm, malloced, hot, submitted, complete, warm2, warm2freed, cool2;
};

uint64_t nanos(std::chrono::high_resolution_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}


void loadmodel() {
    clockwork::tvmutil::initializeTVMCudaStream();
    clockwork::util::setCurrentThreadMaxPriority();
    clockwork::util::set_core(7);

	const int dtype_code = kDLFloat;
	const int dtype_bits = 32;
	const int dtype_lanes = 1;
	const int device_type = kDLGPU;
	const int device_id = 0;

	std::string model = "/home/jcmace/modelzoo/resnet18/tesla-m40_batchsize1/resnet18v2-batchsize1-optimized";


    clockwork::model::ColdModel* cold = clockwork::model::FromDisk(
            model + ".so",
            model + ".clockwork",
            model + ".clockwork_params"
        );
    clockwork::model::CoolModel* cool = cold->load();



    void* ptr;

    cudaEvent_t prehot, posthot, postinput, postcall, postoutput;
    CUDA_CALL(cudaEventCreate(&prehot));
    CUDA_CALL(cudaEventCreate(&posthot));
    CUDA_CALL(cudaEventCreate(&postinput));
    CUDA_CALL(cudaEventCreate(&postcall));
    CUDA_CALL(cudaEventCreate(&postoutput));

    cudaStream_t stream = tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream;

    void* input;
    void* output;


    unsigned runs = 1000;
    std::vector<ProfileData> d(runs);
    for (unsigned i = 0; i < runs; i++) {
        if (i % 100 == 0) {
            std::cout << "Run " << i << std::endl;

            size_t free, total;
            CUDA_CALL(cudaMemGetInfo(&free, &total));
            std::cout << "   GPU " << (total-free) << " used" << std::endl;
        }
        d[i].cool = std::chrono::high_resolution_clock::now();

        clockwork::model::WarmModel* warm = cool->load();
        d[i].warm = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            CUDA_CALL(cudaMalloc(&ptr, warm->size()));
        }
        d[i].malloced = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaEventRecord(prehot, stream));
        clockwork::model::HotModel* hot = warm->load(ptr);
        //CUDA_CALL(cudaStreamSynchronize(stream));
        //d[i].hot = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            CUDA_CALL(cudaMallocHost(&input, hot->inputsize()));
            CUDA_CALL(cudaMallocHost(&output, hot->outputsize()));
        }
        CUDA_CALL(cudaEventRecord(posthot, stream));
        hot->setinput(input);

        CUDA_CALL(cudaEventRecord(postinput, stream));
        hot->call();
        //d[i].submitted = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaEventRecord(postcall, stream));
        hot->getoutput(output);
        CUDA_CALL(cudaEventRecord(postoutput, stream));


        CUDA_CALL(cudaStreamSynchronize(stream));
        d[i].complete = std::chrono::high_resolution_clock::now();

        hot->unload();
        d[i].warm2 = std::chrono::high_resolution_clock::now();

        d[i].warm2freed = std::chrono::high_resolution_clock::now();

        warm->unload();
        d[i].cool2 = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaEventElapsedTime(&(d[i].params), prehot, posthot));
        CUDA_CALL(cudaEventElapsedTime(&(d[i].input), posthot, postinput));
        CUDA_CALL(cudaEventElapsedTime(&(d[i].exec), postinput, postcall));
        CUDA_CALL(cudaEventElapsedTime(&(d[i].output), postcall, postoutput));
    }
    CUDA_CALL(cudaFree(ptr));

    std::ofstream out("times.out");
    out << "i" << "\t"
        << "t" << "\t"
        << "cool->warm" << "\t"
        << "warm->malloced" << "\t"
        << "cuda-params" << "\t"
        << "cuda-input" << "\t"
        << "cuda-exec" << "\t"
        << "cuda-output" << "\t"
        << "malloced->complete" << "\t"
        << "complete->warm2" << "\t"
        << "warm2->freed" << "\t"
        << "freed->cool" << "\n";
    for (unsigned i = 10; i < d.size(); i++) {
        out << i << "\t"
            << nanos(d[i].cool) << "\t"
            << nanos(d[i].warm) - nanos(d[i].cool) << "\t"
            << nanos(d[i].malloced) - nanos(d[i].warm) << "\t"
            << uint64_t(d[i].params * 1000000) << "\t"
            << uint64_t(d[i].input * 1000000) << "\t"
            << uint64_t(d[i].exec * 1000000) << "\t"
            << uint64_t(d[i].output * 1000000) << "\t"
            << nanos(d[i].complete) - nanos(d[i].malloced) << "\t"
            << nanos(d[i].warm2) - nanos(d[i].complete) << "\t"
            << nanos(d[i].warm2freed) - nanos(d[i].warm2) << "\t"
            << nanos(d[i].cool2) - nanos(d[i].warm2freed) << "\n";
    }
    out.close();
}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

	loadmodel();

	std::cout << "end" << std::endl;
}
