#define CUDA_API_PER_THREAD_DEFAULT_STREAM


#include <condition_variable>
#include <iostream>
#include <sstream>
#include <atomic>
#include <thread>
#include <fstream>
#include <istream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mutex>
#include <chrono>
#include "clockwork/util.h"
#include <tvm/runtime/cuda_common.h>
#include "clockwork/cache.h"
#include <dmlc/logging.h>
#include "clockwork/alternatives.h"
#include "clockwork/alternatives/worker.h"
#include <future>
#include <unordered_map>
#include <chrono>
#include "clockwork/telemetry.h"
#include "clockwork/model/model.h"
#include "clockwork/api/api_common.h"
#include "clockwork/api/client_api.h"
#include "clockwork/alternatives/runtime_model.h"

using namespace clockwork;

template<typename R>
  bool is_ready(std::shared_future<R> const& f)
  { return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

struct ModelInfo {
	int id;
	size_t input_size;
	size_t output_size;
	char* input;
	std::string source;
};

class LoadedModels {
private:
	std::unordered_map<int, ModelInfo> model_infos;

public:
	clockwork::alternatives::Worker* worker;

	int user_id = 0;
	std::atomic_int request_id_seed;
	std::atomic_int num_pending;

	LoadedModels(clockwork::alternatives::Worker* worker) : worker(worker), num_pending(0), request_id_seed(0) {}

	~LoadedModels() {
		for (auto &p : model_infos) {
			free(p.second.input);
		}
	}

	void checkHeader(clockwork::ResponseHeader &header) {
		CHECK(header.status == clockworkSuccess) << "Error from worker: " << header.message;
	}

	void checkResponse(int model_id, clockwork::clientapi::InferenceResponse &rsp) {
		checkHeader(rsp.header);
		CHECK(model_infos[model_id].output_size == rsp.output_size);
	}

	int load(std::string model_path) {
		clockwork::clientapi::LoadModelFromRemoteDiskRequest request{
			clockwork::RequestHeader{user_id, request_id_seed++},
			model_path
		};

		// Expected to be synchronous
		worker->loadRemoteModel(request, [this, model_path](clockwork::clientapi::LoadModelFromRemoteDiskResponse& response) {
			this->checkHeader(response.header);

			ModelInfo info{
				response.model_id,
				response.input_size,
				response.output_size,
				static_cast<char*>(malloc(response.input_size)),
				model_path
			};

			model_infos[response.model_id] = info;
		});
	}

	void evict(int model_id) {
		clockwork::clientapi::EvictRequest request {
			clockwork::RequestHeader{user_id, request_id_seed++},
			model_id
		};

		// Expected to be synchronous
		worker->evict(request, [this](clockwork::clientapi::EvictResponse& response) {
			this->checkHeader(response.header);
		});
	}

	void infer(int model_id) {
		ModelInfo info = model_infos[model_id];

		clockwork::clientapi::InferenceRequest request{
			clockwork::RequestHeader{user_id, request_id_seed++},
			model_id,
			1,
			info.input_size,
			info.input
		};

		this->num_pending++;

		worker->infer(request, [this, model_id](clockwork::clientapi::InferenceResponse &response){
			this->checkResponse(model_id, response);
			this->num_pending--;
		});
	}

	void await(int max_pending) {
		while (num_pending >= max_pending) {}
	}

};

clockwork::alternatives::Worker* createWorker(size_t cache_size, size_t page_size, std::string telemetry_filename) {
	
	alternatives::Runtime* runtime = clockwork::alternatives::newGreedyRuntime(1, 1);

	void* baseptr;
	CUDA_CALL(cudaMalloc(&baseptr, cache_size));
	PageCache* cache = new PageCache(static_cast<char*>(baseptr), cache_size, page_size);

	TelemetryLogger* logger = new TelemetryFileLogger(telemetry_filename);

	return new clockwork::alternatives::Worker(runtime, cache, logger);
}

void testmemory(uint64_t cache_size, uint64_t page_size, std::string model_filename, std::string telemetry_filename) {

	clockwork::alternatives::Worker* worker = createWorker(cache_size, page_size, telemetry_filename);

	LoadedModels models(worker);

	int hot_models = 1;
	int num_models = 10;
	for (unsigned i = 0; i < num_models; i++) {
		models.load(model_filename);
		std::cout << "Loading... " << (num_models - i) << " \r";
		std::cout.flush();
	}
	std::cout << "Loaded " << num_models << " models" << std::endl;

	int iterations = 1000;
	int max_outstanding = 4;

	uint64_t last_print = 0;

	for (unsigned i = 0; i < iterations; i++) {
		// Do a hot model
		{
			int next = rand() % hot_models;
			models.infer(next);
			models.await(max_outstanding);
		}
		// Do a random one
		{
			int next = hot_models + (rand() % (num_models - hot_models));
			models.infer(next);
			models.await(max_outstanding);
		}
		// models.evict(j);

		uint64_t now = clockwork::util::now();
		if ((now - last_print) > 100000000L) {
			last_print = now;
			std::cout << (iterations - i) << " \r";
			std::cout.flush();
		}
	}

	models.await(1);
	worker->shutdown();
	delete worker;
}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

    uint64_t page_size = 16L * 1024 * 1024;
    uint64_t cache_size = 16L * 50 * 1024 * 1024;
    std::string model_filename = "/home/jcmace/modelzoo/resnet50/tesla-m40-2_batchsize1/tvm-model";
    std::string telemetry_filename = "telemetry.out";

	testmemory(cache_size, page_size, model_filename, telemetry_filename);

	std::cout << "end" << std::endl;
}
