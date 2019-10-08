#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "clockwork/common.h"
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
#include <cuda_runtime.h>
#include <chrono>
#include <tvm/runtime/cuda_common.h>
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/util.h"


struct ProfileData {
    float weights, input, exec, output;
    std::chrono::high_resolution_clock::time_point start, on_host, on_device, weights_on_device, inputs_on_device, call, outputs_from_device, off_device, off_host;
};

uint64_t nanos(std::chrono::high_resolution_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}


void loadmodel(std::string model_filename) {
    clockwork::util::initializeCudaStream();
    clockwork::util::setCurrentThreadMaxPriority();
    clockwork::util::set_core(7);


    clockwork::model::Model* model = clockwork::model::Model::loadFromDisk(
            model_filename + ".so",
            model_filename + ".clockwork",
            model_filename + ".clockwork_params"
        );


    int page_size = 16 * 1024 * 1024;
    std::vector<char*> weights_pages;
    std::vector<char*> workspace_pages;

    cudaEvent_t start, weights, input, exec, output;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&weights));
    CUDA_CALL(cudaEventCreate(&input));
    CUDA_CALL(cudaEventCreate(&exec));
    CUDA_CALL(cudaEventCreate(&output));

    cudaStream_t stream = tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream;

    void* input_ptr;
    void* output_ptr;

    unsigned runs = 1000;
    std::vector<ProfileData> d(runs);
    for (unsigned i = 0; i < runs; i++) {
        if (i % 100 == 0) {
            std::cout << "Run " << i << std::endl;

            size_t free, total;
            CUDA_CALL(cudaMemGetInfo(&free, &total));
            std::cout << "   GPU " << (total-free) << " used" << std::endl;
        }

        d[i].start = std::chrono::high_resolution_clock::now();

        model->instantiate_model_on_host();

        d[i].on_host = std::chrono::high_resolution_clock::now();
        
        model->instantiate_model_on_device();

        d[i].on_device = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            for (unsigned j = 0; j < model->num_weights_pages(page_size); j++) {
                void* ptr;
                CUDA_CALL(cudaMalloc(&ptr, page_size));
                weights_pages.push_back(static_cast<char*>(ptr));
            }
            for (unsigned j = 0; j < model->num_workspace_pages(page_size); j++) {
                void* ptr;
                CUDA_CALL(cudaMalloc(&ptr, page_size));
                workspace_pages.push_back(static_cast<char*>(ptr));
            }
            CUDA_CALL(cudaMallocHost(&input_ptr, model->input_size()));
            CUDA_CALL(cudaMallocHost(&output_ptr, model->output_size()));
        }

        CUDA_CALL(cudaEventRecord(start, stream));

        model->transfer_weights_to_device(weights_pages, stream);

        CUDA_CALL(cudaEventRecord(weights, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
        d[i].weights_on_device = std::chrono::high_resolution_clock::now();

        model->transfer_input_to_device(static_cast<char*>(input_ptr), workspace_pages, stream);

        CUDA_CALL(cudaEventRecord(input, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
        d[i].inputs_on_device = std::chrono::high_resolution_clock::now();

        model->call(weights_pages, workspace_pages, stream);

        CUDA_CALL(cudaEventRecord(exec, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
        d[i].call = std::chrono::high_resolution_clock::now();

        model->transfer_output_from_device(static_cast<char*>(output_ptr), workspace_pages, stream);

        CUDA_CALL(cudaEventRecord(output, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
        d[i].outputs_from_device = std::chrono::high_resolution_clock::now();

        model->uninstantiate_model_on_device();

        d[i].off_device = std::chrono::high_resolution_clock::now();

        model->uninstantiate_model_on_host();

        d[i].off_host = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaEventElapsedTime(&(d[i].weights), start, weights));
        CUDA_CALL(cudaEventElapsedTime(&(d[i].input), weights, input));
        CUDA_CALL(cudaEventElapsedTime(&(d[i].exec), input, exec));
        CUDA_CALL(cudaEventElapsedTime(&(d[i].output), exec, output));
    }

    for (char* &ptr : weights_pages) {
        CUDA_CALL(cudaFree(ptr));
    }
    for (char* &ptr : workspace_pages) {
        CUDA_CALL(cudaFree(ptr));
    }


struct ProfileData {
    float weights, input, exec, output;
    std::chrono::high_resolution_clock::time_point start, on_host, on_device, weights_on_device, inputs_on_device, call, outputs_from_device, off_device, off_host;
};

    std::ofstream out("times.out");
    out << "i" << "\t"
        << "t" << "\t"
        << "load_host_side" << "\t"
        << "load_device_code" << "\t"
        << "transfer_weights" << "\t"
        << "transfer_weights (cuda)" << "\t"
        << "transfer_input" << "\t"
        << "transfer_input (cuda)" << "\t"
        << "run" << "\t"
        << "run (cuda)" << "\t"
        << "transfer_output" << "\t"
        << "transfer_output (cuda)" << "\t"
        << "unload_device_code" << "\t"
        << "unload_host_side" << "\n";
    for (unsigned i = 10; i < d.size(); i++) {
        out << i << "\t"
            << nanos(d[i].start) << "\t"
            << nanos(d[i].on_host) - nanos(d[i].start) << "\t"
            << nanos(d[i].on_device) - nanos(d[i].on_host) << "\t"
            << nanos(d[i].weights_on_device) - nanos(d[i].on_device) << "\t"
            << uint64_t(d[i].weights * 1000000) << "\t"
            << nanos(d[i].inputs_on_device) - nanos(d[i].weights_on_device) << "\t"
            << uint64_t(d[i].input * 1000000) << "\t"
            << nanos(d[i].call) - nanos(d[i].inputs_on_device) << "\t"
            << uint64_t(d[i].exec * 1000000) << "\t"
            << nanos(d[i].outputs_from_device) - nanos(d[i].call) << "\t"
            << uint64_t(d[i].output * 1000000) << "\t"
            << nanos(d[i].off_device) - nanos(d[i].outputs_from_device) << "\t"
            << nanos(d[i].off_host) - nanos(d[i].off_device) << "\n";
    }
    out.close();
}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

    if (argc != 2) {
        std::cout << "Expected a model as input" << std::endl;
        return -1;
    }

	loadmodel(argv[1]);

	std::cout << "end" << std::endl;
}
