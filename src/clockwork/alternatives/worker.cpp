#include "clockwork/alternatives/worker.h"
#include <iostream>
#include <atomic>
#include "clockwork/util.h"
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"

using namespace clockwork::alternatives;

ModelManager::ModelManager(const int id, Runtime* runtime, PageCache* cache, model::ColdModel* cold) : id(id), runtime(runtime), model(cache, cold) {
	model.coldToCool();
	model.coolToWarm();
}

std::atomic_int request_id_seed = 0;

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

	queue_mutex.lock();
	pending_requests.push_back(r);
	if (!in_use.test_and_set()) {
		Request* toSubmit = pending_requests.front();
		pending_requests.pop_front();
		queue_mutex.unlock();
		submit(toSubmit);
	} else {
		queue_mutex.unlock();
	}

	return std::shared_future<InferenceResponse>(r->promise.get_future());
}


void ModelManager::handle_response(Request* request) {
	request->promise.set_value(
		InferenceResponse{
			ResponseHeader{clockworkSuccess, ""}, 
			model.outputsize(), 
			request->output
		});
	delete request;

	queue_mutex.lock();
	if (pending_requests.size() > 0) {
		Request* toSubmit = pending_requests.front();
		pending_requests.pop_front();
		queue_mutex.unlock();
		submit(toSubmit);
	} else {
		model.unlock();
		in_use.clear();
		queue_mutex.unlock();
	}
}

void ModelManager::submit(Request* request) {
	RuntimeModel::State state = model.lock();


	RequestBuilder* builder = runtime->newRequest();

	if (state == RuntimeModel::State::Warm) {
		builder->addTask(TaskType::PCIe_H2D_Weights, [this] {
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
	builder->addTask(TaskType::Sync, [this, request] {
		// cudaStreamSynchronize might not be necessary -- it waits for the PCIe_D2H_Output to complete,
		// but some executor types might already guarantee it's completed.  Some, however, will not
		// provide this guarantee, and only do a cudaStreamWaitEvent on the current stream.
		CUDA_CALL(cudaStreamSynchronize(util::Stream()));
		this->handle_response(request);
	});

	builder->submit();
}

Worker::Worker(Runtime* runtime, PageCache* cache) : runtime(runtime), cache(cache) {}

std::shared_future<LoadModelFromDiskResponse> Worker::loadModelFromDisk(LoadModelFromDiskRequest &request) {
	// Synchronous for now since this is not on critical path
	std::lock_guard<std::mutex> lock(managers_mutex);

	clockwork::model::ColdModel* cold = clockwork::model::FromDisk(
		request.model_path + ".so",
		request.model_path + ".clockwork",
		request.model_path + ".clockwork_params"
	);
	int id = managers.size();
	ModelManager* manager = new ModelManager(id, runtime, cache, cold);
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