#include "clockwork/worker.h"
#include "clockwork/network/worker.h"
#include "clockwork/runtime.h"
#include "clockwork/model/batched.h"
#include "clockwork/memory.h"
#include "clockwork/cache.h"
#include "clockwork/cuda_common.h"

using namespace clockwork;


void show_usage() {
	std::cout << "USAGE" << std::endl;
	std::cout << "  ./check_model [MODEL]" << std::endl;
	std::cout << "DESCRIPTION" << std::endl;
	std::cout << "  Will load and run an inference on a specified Clockwork model" << std::endl;
	std::cout << "OPTIONS" << std::endl;
    std::cout << "  -h, --help" << std::endl;
    std::cout << "      Print this message" << std::endl;
    std::cout << "  -p, --page_size" << std::endl;
    std::cout << "      Weights page size used by Clockwork.  Defaults to 16MB.  You shouldn't" << std::endl;
    std::cout << "      need to set this because we are using 16MB pages." << std::endl;
}

model::BatchedModel* load_model(std::string model) {
	return model::BatchedModel::loadFromDisk(model, 0);
}

void check_model(int page_size, std::string model_path) {
	std::cout << "Checking " << model_path << std::endl;

	util::setCudaFlags();
    util::initializeCudaStream();

	clockwork::model::BatchedModel* model = load_model(model_path);

	auto batch_sizes = model->implemented_batch_sizes();
	for (unsigned batch_size : batch_sizes) {
		std::cout << "  loaded batch size " << batch_size << std::endl;
	}

    model->instantiate_models_on_host();

    size_t weights_page_size = page_size;
    size_t weights_cache_size = model->num_weights_pages(weights_page_size) * weights_page_size;
    PageCache* weights_cache = make_GPU_cache(weights_cache_size, weights_page_size, GPU_ID_0);

    cudaError_t status;
    model->instantiate_models_on_device();
	    
    std::shared_ptr<Allocation> weights = weights_cache->alloc(model->num_weights_pages(weights_page_size), []{});
    model->transfer_weights_to_device(weights->page_pointers, util::Stream());

    for (unsigned batch_size : batch_sizes) {
    	// Create inputs and outputs
	    char* input = new char[model->input_size(batch_size)];
	    char* output = new char[model->output_size(batch_size)];

	    // Create and allocate io_memory on GPU
    	size_t io_memory_size = model->io_memory_size(batch_size);
    	MemoryPool* io_pool = CUDAMemoryPool::create(io_memory_size, GPU_ID_0);
	    char* io_memory = io_pool->alloc(io_memory_size);

	    // Create and allocate intermediate GPU memory workspace
	    size_t workspace_size = model->workspace_memory_size(batch_size);
    	MemoryPool* workspace_pool = CUDAMemoryPool::create(workspace_size, GPU_ID_0);
	    char* workspace_memory = workspace_pool->alloc(workspace_size);

	    // Now execute each step of model
	    model->transfer_input_to_device(batch_size, input, io_memory, util::Stream());

	    // Time the call
	    int warmups = 20;
		for (int i = 0; i < warmups; i++) {    
	    	model->call(batch_size, weights->page_pointers, io_memory, workspace_memory, util::Stream());
	    }
            status = cudaStreamSynchronize(util::Stream());
            CHECK(status == cudaSuccess);
	    auto before = util::now();
            int iterations = 100;
		for (int i = 0; i < iterations; i++) {    
	    	model->call(batch_size, weights->page_pointers, io_memory, workspace_memory, util::Stream());
	    }
            status = cudaStreamSynchronize(util::Stream());
            CHECK(status == cudaSuccess);
	    auto after = util::now();
	    printf("  b%d: %.2f ms per call\n", batch_size, ((float) (after-before)) / (iterations * 1000000.0));

	    model->transfer_output_from_device(batch_size, output, io_memory, util::Stream());

	    status = cudaStreamSynchronize(util::Stream());
	    CHECK(status == cudaSuccess);

	    delete input;
	    delete output;

	    io_pool->free(io_memory);
    	delete io_pool;

	    workspace_pool->free(workspace_memory);
    	delete workspace_pool;
	}

    weights_cache->unlock(weights);
    weights_cache->free(weights);
    delete weights_cache;

    model->uninstantiate_models_on_device();
    model->uninstantiate_models_on_host();

    delete model;
}

int main(int argc, char *argv[]) {
	std::vector<std::string> non_argument_strings;

	int page_size = 16 * 1024 * 1024;
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if ((arg == "-h") || (arg == "--help")) {
		    show_usage();
		    return 0;
		} else if ((arg == "-p") || (arg == "--page_size")) {
		    page_size = atoi(argv[++i]);
		} else {
		 	non_argument_strings.push_back(arg);
		}
	}

	if (non_argument_strings.size() != 1) {
		std::cerr << "Expecting a model as input" << std::endl;
		return 1;
	}

	std::string model_path = non_argument_strings[0];

	check_model(page_size, model_path);
}
