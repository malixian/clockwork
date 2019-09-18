#include "clockwork/alternatives/worker.h"
#include <iostream>
#include <atomic>
#include "clockwork/util.h"
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/telemetry.h"

using namespace clockwork::alternatives;

ModelManager::ModelManager(const int id, Runtime* runtime, PageCache* cache, model::Model* model, TelemetryLogger* logger) : id(id), runtime(runtime), model(cache, model), logger(logger), request_id_seed(0) {
}

EvictResponse ModelManager::evict() {
	std::lock_guard<std::mutex> lock(queue_mutex);

	if (!model.try_lock()) {
		std::stringstream errorMsg;
		errorMsg << "Cannot evict model that is in use, model_id=" << id;
		return EvictResponse{ResponseHeader{clockworkError, errorMsg.str()}};	
	}

	model.evict_weights();
	model.unlock();

	return EvictResponse{ResponseHeader{clockworkSuccess, ""}};
}


std::shared_future<InferenceResponse> ModelManager::add_request(InferenceRequest &request) {

	if (request.input_size != model.input_size()) {
		std::stringstream errorMsg;
		errorMsg << "Mismatched input size, expected " << model.input_size() << ", got " << request.input_size;

		std::promise<InferenceResponse> response;
		response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
		return std::shared_future<InferenceResponse>(response.get_future());
	}

	if (request.output_size != model.output_size()) {
		std::stringstream errorMsg;
		errorMsg << "Mismatched input size, expected " << model.output_size() << ", got " << request.output_size;

		std::promise<InferenceResponse> response;
		response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
		return std::shared_future<InferenceResponse>(response.get_future());
	}

	Request* r = new Request();
	r->id = request_id_seed++;
	r->input = request.input;
	r->output = request.output; // Later don't use malloc?

	r->telemetry = new RequestTelemetry();
	r->telemetry->model_id = id;
	r->telemetry->arrived = clockwork::util::hrt();

	Request* toSubmit = nullptr;
	{
		std::lock_guard<std::mutex> lock(queue_mutex);

		pending_requests.push_back(r);
		if (model.try_lock()) {
			toSubmit = pending_requests.front();
			pending_requests.pop_front();
		}
	}

	if (toSubmit != nullptr) {
		submit(toSubmit);
	}
	

	return std::shared_future<InferenceResponse>(r->promise.get_future());
}

void ModelManager::submit(Request* request) {
	RequestBuilder* builder = runtime->newRequest();

	builder->setTelemetry(request->telemetry);
	
	if (!model.has_code()) {
		builder->addTask(TaskType::ModuleLoad, [this] {
			this->model.instantiate_code();
		});
	}

	if (!model.has_weights()) {
		builder->addTask(TaskType::PCIe_H2D_Weights, [this] {
			this->model.transfer_weights(util::Stream()); // TODO: pass stream as argument to function
		});
	}

	builder->addTask(TaskType::PCIe_H2D_Inputs, [this, request] {
		this->model.set_input(request->input, util::Stream());
    });

	builder->addTask(TaskType::GPU, [this] {
		this->model.call(util::Stream());
	});

	builder->addTask(TaskType::PCIe_D2H_Output, [this, request] {
		this->model.get_output(request->output, util::Stream());
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


void ModelManager::handle_response(Request* request) {
	request->telemetry->complete = clockwork::util::hrt();

	Request* toSubmit = nullptr;
	{
		std::lock_guard<std::mutex> lock(queue_mutex);

		if (pending_requests.size() > 0) {
			toSubmit = pending_requests.front();
			pending_requests.pop_front();
		}
	}

	if (toSubmit != nullptr) {
		submit(toSubmit);
	} else {
		model.unlock();
	}

	request->promise.set_value(
		InferenceResponse{
			ResponseHeader{clockworkSuccess, ""}, 
			model.output_size(), 
			request->output
		});
	this->logger->log(request->telemetry);
	delete request;
}

Worker::Worker(Runtime* runtime, PageCache* cache, TelemetryLogger *logger) : runtime(runtime), cache(cache), logger(logger) {}

std::shared_future<LoadModelFromDiskResponse> Worker::loadModelFromDisk(LoadModelFromDiskRequest &request) {
	// Synchronous for now since this is not on critical path
	std::lock_guard<std::mutex> lock(managers_mutex);

	model::Model* model = model::Model::loadFromDisk(
		request.model_path + ".so",
		request.model_path + ".clockwork",
		request.model_path + ".clockwork_params"
	);
	int id = managers.size();
	ModelManager* manager = new ModelManager(id, runtime, cache, model, logger);
	managers.push_back(manager);


	std::promise<LoadModelFromDiskResponse> response;
	response.set_value(
		LoadModelFromDiskResponse{
			ResponseHeader{clockworkSuccess, ""},
			id,
			manager->model.input_size(),
			manager->model.output_size()
		});
	return std::shared_future<LoadModelFromDiskResponse>(response.get_future());
}

std::shared_future<InferenceResponse> Worker::infer(InferenceRequest &request) {
	ModelManager* manager = nullptr;
	{
		std::lock_guard<std::mutex> lock(managers_mutex);

		if (request.model_id < 0 || request.model_id >= managers.size()) {
			std::stringstream errorMsg;
			errorMsg << "No model exists with ID " << request.model_id;

			std::promise<InferenceResponse> response;
			response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
			return std::shared_future<InferenceResponse>(response.get_future());
		}
		manager = managers[request.model_id];
	}
	return manager->add_request(request);
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
