#ifndef _CLOCKWORK_ALTERNATIVES_WORKER_H_
#define _CLOCKWORK_ALTERNATIVES_WORKER_H_

#include <mutex>
#include <deque>
#include <unordered_map>
#include <future>
#include <sstream>

#include "clockwork/runtime.h"
#include "clockwork/runtime_model.h"
#include "clockwork/cache.h"
#include "clockwork/model.h"


const int clockworkSuccess = 0;
const int clockworkError = 1;

namespace clockwork {
namespace alternatives {

struct RequestHeader {
	// Placeholder
};

struct ResponseHeader {
	int status;
	std::string message;
};

/**
TODO: loading model across network not included
**/

struct LoadModelFromDiskRequest {
	RequestHeader header;
	std::string model_path;
};

struct LoadModelFromDiskResponse {
	ResponseHeader header;
	int model_id;
	int input_size;
};

struct InferenceRequest {
	RequestHeader header;
	int model_id;
	int input_size;
	char* input;
};

struct InferenceResponse {
	ResponseHeader header;
	int output_size;
	char* output;
};


/** Manages a specific model instance */
class ModelManager {
public:
	struct Request {
		unsigned id;
		char* input;
		char* output;
		std::promise<InferenceResponse> promise;
	};

	const int id;
	Runtime* runtime;

	// The model being managed
	RuntimeModel model;

	std::mutex queue_mutex;
	std::deque<Request*> pending_requests;
	std::atomic_flag in_use = ATOMIC_FLAG_INIT;	// Only one request can execute at a time for a model

	ModelManager(const int id, Runtime* runtime, PageCache* cache, model::ColdModel* cold);
	std::shared_future<InferenceResponse> add_request(InferenceRequest &request);

private:

	void handle_response(Request* request);
	void submit(Request* request);

};

/** The opposite of clockwork; manages everything itself */
class Worker {
private:
	Runtime* runtime;
	PageCache* cache;

	std::mutex managers_mutex;
	std::vector<ModelManager*> managers;

public:

	Worker(Runtime* runtime, PageCache* cache);
	std::shared_future<LoadModelFromDiskResponse> loadModelFromDisk(LoadModelFromDiskRequest &request);
	std::shared_future<InferenceResponse> infer(InferenceRequest &request);
};

}
}

#endif