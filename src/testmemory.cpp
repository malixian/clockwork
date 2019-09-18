#define CUDA_API_PER_THREAD_DEFAULT_STREAM


#include <iostream>
#include <sstream>
#include <atomic>
#include <thread>
#include <fstream>
#include <istream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include "clockwork/util.h"
#include <tvm/runtime/cuda_common.h>
#include "clockwork/cache.h"
#include <dmlc/logging.h>
#include "clockwork/runtime.h"
#include "clockwork/alternatives/worker.h"
#include <future>
#include <unordered_map>
#include <chrono>
#include "clockwork/telemetry.h"
#include "clockwork/model/model.h"
#include "clockwork/runtime_model.h"

using namespace clockwork;

template<typename R>
  bool is_ready(std::shared_future<R> const& f)
  { return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

struct ModelInfo {
	int id;
	int input_size;
	int output_size;
	char* input;
	char* output;
	std::string source;
};

class LoadedModels {
public:
	clockwork::alternatives::Worker* worker;
	std::unordered_map<int, ModelInfo> model_infos;

	std::vector<std::shared_future<clockwork::alternatives::InferenceResponse>> pending;

	LoadedModels(clockwork::alternatives::Worker* worker) : worker(worker) {}

	void checkHeader(clockwork::alternatives::ResponseHeader header) {
		CHECK(header.status == clockworkSuccess) << "Error inferring model: " << header.message;
	}

	void checkResponse(clockwork::alternatives::InferenceResponse rsp) {
		checkHeader(rsp.header);
		// std::cout << "Got response of size " << rsp.output_size << std::endl;
	}

	int load(std::string modelPath) {
		clockwork::alternatives::LoadModelFromDiskRequest request;
		request.model_path = modelPath;
		auto rsp = worker->loadModelFromDisk(request).get();
		CHECK(rsp.header.status == clockworkSuccess) << "Error loading model: " << rsp.header.message;
		// std::cout << "Loaded " << rsp.model_id << ": [" << rsp.input_size << "] " << modelPath << std::endl;
		
		ModelInfo info;
		info.id = rsp.model_id;
		info.input_size = rsp.input_size;
		info.output_size = rsp.output_size;
		CUDA_CALL(cudaMallocHost(&info.input, info.input_size));
		CUDA_CALL(cudaMallocHost(&info.output, info.output_size));
		info.source = modelPath;

		model_infos[rsp.model_id] = info;

		return rsp.model_id;
	}

	void evict(int model_id) {
		clockwork::alternatives::EvictRequest request;
		request.model_id = model_id;
		auto f = worker->evict(request);
		auto rsp = f.get();
		checkHeader(rsp.header);
	}

	std::shared_future<clockwork::alternatives::InferenceResponse> infer(int model_id) {
		// std::cout << "Inferring on " << model_id << " with input size " << sizes[model_id] << std::endl;
		
		ModelInfo info = model_infos[model_id];

		clockwork::alternatives::InferenceRequest request;
		request.model_id = model_id;
		request.input_size = info.input_size;
		request.input = info.input;
		request.output_size = info.output_size;
		request.output = info.output;
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
};

void testmemory(uint64_t totalsize, uint64_t pagesize) {
	
	Runtime* runtime = clockwork::newGreedyRuntime(1, 1);

	void* baseptr;
	CUDA_CALL(cudaMalloc(&baseptr, totalsize));
	PageCache* cache = new PageCache(static_cast<char*>(baseptr), totalsize, pagesize);

	TelemetryLogger* logger = new TelemetryLogger("telemetry.out");

	clockwork::alternatives::Worker* worker = new clockwork::alternatives::Worker(runtime, cache, logger);

	LoadedModels models(worker);

	int hot_models = 1;
	int num_models = 10;
	for (unsigned i = 0; i < num_models; i++) {
		models.load("/home/jcmace/modelzoo/resnet50/tesla-m40-2_batchsize1/tvm-model");
		//models.load("/home/jcmace/modelzoo/resnet18/tesla-m40-2_batchsize1/tvm-model");
	}

	std::cout << "Loaded " << num_models << " models" << std::endl;

	int iterations = 10000;
	int max_outstanding = 4;
	for (unsigned i = 0; i < iterations; i++) {
		// Do a hot model
		{
			int next = rand() % hot_models;
			models.infer(next);
			while (models.pending.size() >= max_outstanding) {
				models.checkResponse(models.awaitOne().get());						
			}
		}
		// Do a random one
		{
			int next = hot_models + (rand() % (num_models - hot_models));
			models.infer(next);
			while (models.pending.size() >= max_outstanding) {
				models.checkResponse(models.awaitOne().get());						
			}
		}
		// models.evict(j);
	}

}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

    uint64_t pagesize = 16L * 1024 * 1024;
    uint64_t totalsize = 16L * 50 * 1024 * 1024;
	testmemory(totalsize, pagesize);

	std::cout << "end" << std::endl;
}
