#include "clockwork/alternatives/worker.h"
#include <iostream>
#include <atomic>
#include "clockwork/util.h"
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/telemetry.h"

using namespace clockwork::alternatives;

ModelManager::ModelManager(const int id, Runtime* runtime, PageCache* cache, model::ColdModel* cold, TelemetryLogger* logger) : id(id), runtime(runtime), model(cache, cold), logger(logger) {
	model.coldToCool();
	model.coolToWarm();
}

std::atomic_int request_id_seed = 0;

EvictResponse ModelManager::evict() {
	std::lock_guard<std::mutex> lock(queue_mutex);

	if (in_use.test_and_set()) {
		std::stringstream errorMsg;
		errorMsg << "Cannot evict model that is in use, model_id=" << id;
		return EvictResponse{ResponseHeader{clockworkError, errorMsg.str()}};	
	}

	model.evict();
	in_use.clear();

	return EvictResponse{ResponseHeader{clockworkSuccess, ""}};
}

std::shared_future<InferenceResponse> ModelManager::add_request(InferenceRequest &request) {

	if (request.input_size != model.inputsize()) {
		std::stringstream errorMsg;
		errorMsg << "Mismatched input size, expected " << model.inputsize() << ", got " << request.input_size;

		std::promise<InferenceResponse> response;
		response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
		return std::shared_future<InferenceResponse>(response.get_future());
	}

	Request* r = new Request();
	r->id = request_id_seed++;
	r->input = request.input;
	r->output = static_cast<char*>(malloc(model.outputsize())); // Later don't use malloc?

	r->telemetry = new RequestTelemetry();
	r->telemetry->model_id = id;
	r->telemetry->arrived = clockwork::util::hrt();

	Request* toSubmit = nullptr;
	{
		std::lock_guard<std::mutex> lock(queue_mutex);

		pending_requests.push_back(r);
		if (!in_use.test_and_set()) {
			toSubmit = pending_requests.front();
			pending_requests.pop_front();
		}
	}

	if (toSubmit != nullptr) {
		submit(toSubmit);
	}
	

	return std::shared_future<InferenceResponse>(r->promise.get_future());
}


void ModelManager::handle_response(Request* request) {
	request->telemetry->complete = clockwork::util::hrt();

	model.unlock();

	Request* toSubmit = nullptr;
	{
		std::lock_guard<std::mutex> lock(queue_mutex);

		if (pending_requests.size() > 0) {
			toSubmit = pending_requests.front();
			pending_requests.pop_front();
		} else {
			in_use.clear();
		}
	}

	if (toSubmit != nullptr) {
		submit(toSubmit);
	}

	request->promise.set_value(
		InferenceResponse{
			ResponseHeader{clockworkSuccess, ""}, 
			model.outputsize(), 
			request->output
		});
	this->logger->log(request->telemetry);
	delete request;
}

void ModelManager::submit(Request* request) {
	RuntimeModel::State state = model.lock();

	RequestBuilder* builder = runtime->newRequest();

	builder->setTelemetry(request->telemetry);
	
	if (state == RuntimeModel::State::Warm) {
		builder->addTask(TaskType::PCIe_H2D_Weights, [this, request] {
			this->model.warmToHot();
	    });
	}

	if (state == RuntimeModel::State::Exec) {
    	builder->addTask(TaskType::PCIe_H2D_Inputs, [this, request] {
    		this->model.setInput(request->input);
    	});
	} else {
    	builder->addTask(TaskType::PCIe_H2D_Inputs, [this, request] {
    		this->model.hotToExec();
    		this->model.setInput(request->input);
    	});
	}

	builder->addTask(TaskType::GPU, [this] {
		this->model.call();
	});

	builder->addTask(TaskType::PCIe_D2H_Output, [this, request] {
		this->model.getOutput(request->output);
	});

	// Task is unnecessary since onComplete callback won't run until async part of previous task is completed
	// builder->addTask(TaskType::Sync, [this, request] {
	// 	// cudaStreamSynchronize might not be necessary -- it waits for the PCIe_D2H_Output to complete,
	// 	// but some executor types might already guarantee it's completed.  Some, however, will not
	// 	// provide this guarantee, and only do a cudaStreamWaitEvent on the current stream.
	// 	CUDA_CALL(cudaStreamSynchronize(util::Stream()));
	// });

	builder->setCompletionCallback([this, request] {
		this->handle_response(request);
	});

	request->telemetry->submitted = clockwork::util::hrt();

	builder->submit();
}

Worker::Worker(Runtime* runtime, PageCache* cache, TelemetryLogger *logger) : runtime(runtime), cache(cache), logger(logger) {}

std::shared_future<LoadModelFromDiskResponse> Worker::loadModelFromDisk(LoadModelFromDiskRequest &request) {
	// Synchronous for now since this is not on critical path
	std::lock_guard<std::mutex> lock(managers_mutex);

	clockwork::model::ColdModel* cold = clockwork::model::FromDisk(
		request.model_path + ".so",
		request.model_path + ".clockwork",
		request.model_path + ".clockwork_params"
	);
	int id = managers.size();
	ModelManager* manager = new ModelManager(id, runtime, cache, cold, logger);
	managers.push_back(manager);


	std::promise<LoadModelFromDiskResponse> response;
	response.set_value(
		LoadModelFromDiskResponse{
			ResponseHeader{clockworkSuccess, ""},
			id,
			manager->model.inputsize()
		});
	return std::shared_future<LoadModelFromDiskResponse>(response.get_future());
}

std::shared_future<InferenceResponse> Worker::infer(InferenceRequest &request) {
	std::lock_guard<std::mutex> lock(managers_mutex);

	if (request.model_id < 0 || request.model_id >= managers.size()) {
		std::stringstream errorMsg;
		errorMsg << "No model exists with ID " << request.model_id;

		std::promise<InferenceResponse> response;
		response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
		return std::shared_future<InferenceResponse>(response.get_future());
	}
	return managers[request.model_id]->add_request(request);
}

std::shared_future<EvictResponse> Worker::evict(EvictRequest &request) {
	std::promise<EvictResponse> response;

	std::lock_guard<std::mutex> lock(managers_mutex);

	if (request.model_id < 0 || request.model_id >= managers.size()) {
		std::stringstream errorMsg;
		errorMsg << "No model exists with ID " << request.model_id;

		response.set_value(EvictResponse{ResponseHeader{clockworkError, errorMsg.str()}});
	} else {
		response.set_value(managers[request.model_id]->evict());
	}
	return std::shared_future<EvictResponse>(response.get_future());
}