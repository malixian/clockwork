#include "clockwork/alternatives/worker.h"

using namespace clockwork::alternatives;

ModelManager::ModelManager(Runtime* runtime, PageCache* cache, model::ColdModel* cold) : runtime(runtime), model(cache, cold) {
	model.coldToCool();
	model.coolToWarm();
}

std::future<InferenceResponse> ModelManager::add_request(InferenceRequest &request) {
	if (request.input_size != model.inputsize()) {
		std::stringstream errorMsg;
		errorMsg << "Mismatched input size, expected " << model.inputsize() << ", got " << request.input_size;

		std::promise<InferenceResponse> response;
		response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
		return response.get_future();
	}

	Request* r = new Request();
	r->input = request.input;
	r->output = static_cast<char*>(malloc(model.outputsize()));

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

	return r->promise.get_future();
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
		this->handle_response(request);
	});
}

Worker::Worker(Runtime* runtime, PageCache* cache) : runtime(runtime), cache(cache) {}

std::future<LoadModelFromDiskResponse> Worker::loadModelFromDisk(LoadModelFromDiskRequest &request) {
	// Synchronous for now since this is not on critical path
	std::lock_guard<std::mutex> lock(models_mutex);

	clockwork::model::ColdModel* cold = clockwork::model::FromDisk(
		request.model_path + ".so",
		request.model_path + ".clockwork",
		request.model_path + ".clockwork_params"
	);
	ModelManager* manager = new ModelManager(runtime, cache, cold);
	int model_id = model_id_seed++;
	models[model_id] = manager;


	std::promise<LoadModelFromDiskResponse> response;
	response.set_value(
		LoadModelFromDiskResponse{
			ResponseHeader{clockworkSuccess, ""},
			model_id
		});
	return response.get_future();
}

std::future<InferenceResponse> Worker::infer(InferenceRequest &request) {
	auto it = models.find(request.model_id);
	ModelManager* manager;
	if (it == models.end() || (manager = it->second) == nullptr) {
		std::stringstream errorMsg;
		errorMsg << "No model exists with ID " << request.model_id;

		std::promise<InferenceResponse> response;
		response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
		return response.get_future();
	}
	return manager->add_request(request);
}