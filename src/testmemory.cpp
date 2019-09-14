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
#include "clockwork/cache.h"
#include <dmlc/logging.h>
#include "clockwork/clockwork.h"
#include "clockwork/alternatives/worker.h"
#include <future>
#include <unordered_map>
#include <chrono>

using namespace clockwork;

template<typename R>
  bool is_ready(std::shared_future<R> const& f)
  { return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

class LoadedModels {
public:
	clockwork::alternatives::Worker* worker;
	std::unordered_map<int, std::string> sources;
	std::unordered_map<int, int> sizes;
	std::vector<int> model_ids;
	std::vector<std::shared_future<clockwork::alternatives::InferenceResponse>> pending;

	LoadedModels(clockwork::alternatives::Worker* worker) : worker(worker) {}

	void addModel(std::string source, int model_id, int input_size) {
		sources[model_id] = source;
		sizes[model_id] = input_size;
		model_ids.push_back(model_id);
	}

	int load(std::string modelPath) {
		clockwork::alternatives::LoadModelFromDiskRequest request;
		request.model_path = modelPath;
		auto rsp = worker->loadModelFromDisk(request).get();
		CHECK(rsp.header.status == clockworkSuccess) << "Error loading model: " << rsp.header.message;
		// std::cout << "Loaded " << rsp.model_id << ": [" << rsp.input_size << "] " << modelPath << std::endl;
		addModel(modelPath, rsp.model_id, rsp.input_size);
		return rsp.model_id;
	}

	std::shared_future<clockwork::alternatives::InferenceResponse> infer(int model_id) {
		// std::cout << "Inferring on " << model_id << " with input size " << sizes[model_id] << std::endl;
		char* input = static_cast<char*>(malloc(sizes[model_id]));
		clockwork::alternatives::InferenceRequest request;
		request.model_id = model_id;
		request.input_size = sizes[model_id];
		request.input = input;
		auto f = worker->infer(request);
		pending.push_back(f);
		return f;
	}

	std::shared_future<clockwork::alternatives::InferenceResponse> awaitOne() {
		while (true) {
			for (unsigned i = 0; i < pending.size(); i++) {
				auto f = pending[i];
				if (is_ready(f)) {
					pending.erase(pending.begin() + i);
					return f;
				}
			}
		}
	}

	void checkResponse(clockwork::alternatives::InferenceResponse rsp) {
		CHECK(rsp.header.status == clockworkSuccess) << "Error inferring model: " << rsp.header.message;
		// std::cout << "Got response of size " << rsp.output_size << std::endl;
		free(rsp.output);
	}
};

void testmemory(uint64_t totalsize, uint64_t pagesize) {
	
	Runtime* runtime = clockwork::newFIFOThreadpoolRuntime(1);

	void* baseptr;
	CUDA_CALL(cudaMalloc(&baseptr, totalsize));
	PageCache* cache = new PageCache(static_cast<char*>(baseptr), totalsize, pagesize);

	clockwork::alternatives::Worker* worker = new clockwork::alternatives::Worker(runtime, cache);

	LoadedModels models(worker);

	int num_models = 50;
	for (unsigned i = 0; i < num_models; i++) {
		models.load("/home/jcmace/modelzoo/resnet50/tesla-m40_batchsize1/tvm-model");
	}

	std::cout << "Loaded " << num_models << " models" << std::endl;

	int num_per_model = 20;
	for (unsigned i = 0; i < num_per_model; i++) {
		for (unsigned j = 0; j < models.model_ids.size(); j++) {
			models.infer(j);
		}
	}

	while (models.pending.size() > 0) {
		models.checkResponse(models.awaitOne().get());		
	}

}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

    uint64_t pagesize = 16L * 1024 * 1024;
    uint64_t totalsize = 8L * 1024 * 1024 * 1024;
	testmemory(totalsize, pagesize);

	std::cout << "end" << std::endl;
}
