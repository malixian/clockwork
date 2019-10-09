#include "clockwork/worker.h"

namespace clockwork {

// TODO: actually instantiate the clockwork runtime properly and set the controller
ClockworkWorker::ClockworkWorker() : runtime(new ClockworkRuntime()) {
}

ClockworkWorker::~ClockworkWorker() {
    delete runtime;
}

void ClockworkWorker::shutdown(bool await_completion) {
	runtime->shutdown(false);
	if (await_completion) {
		join();
	}
}

void ClockworkWorker::join() {
	runtime->join();
}

void ClockworkWorker::sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions) {
	for (std::shared_ptr<workerapi::Action> action : actions) {
		std::cout << "Received an action " << action->action_type << std::endl;
		switch (action->action_type) {
			case workerapi::loadModelFromDiskAction: loadModel(action); break;
			case workerapi::loadWeightsAction: loadWeights(action); break;
			case workerapi::inferAction: infer(action); break;
			case workerapi::evictWeightsAction: evictWeights(action); break;
			default: invalidAction(action); break;
		}
	}
}

void ClockworkWorker::invalidAction(std::shared_ptr<workerapi::Action> action) {
	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = action->action_type;
	result->status = actionErrorRuntimeError;
	result->message = "Invalid Action";

	controller->sendResult(result);
}

void ClockworkWorker::loadModel(std::shared_ptr<workerapi::Action> action) {
	auto load_model = std::static_pointer_cast<workerapi::LoadModelFromDisk>(action);
	if (load_model != nullptr) {
		LoadModelFromDisk* action = new LoadModelFromDisk(this, load_model);
		action->submit();
	} else {
		invalidAction(action);
	}
}

void ClockworkWorker::loadWeights(std::shared_ptr<workerapi::Action> action) {
	auto load_weights = std::static_pointer_cast<workerapi::LoadWeights>(action);
	if (load_weights != nullptr) {
		LoadWeights* action = new LoadWeights(this, load_weights);
		action->submit();
	} else {
		invalidAction(action);
	}
}

void ClockworkWorker::evictWeights(std::shared_ptr<workerapi::Action> action) {
	auto evict_weights = std::static_pointer_cast<workerapi::EvictWeights>(action);
	if (evict_weights != nullptr) {
		EvictWeights* action = new EvictWeights(this, evict_weights);
		action->submit();
	} else {
		invalidAction(action);
	}
}

void ClockworkWorker::infer(std::shared_ptr<workerapi::Action> action) {
	auto infer = std::static_pointer_cast<workerapi::Infer>(action);
	if (infer != nullptr) {
		Infer* action = new Infer(this, infer);
		action->submit();
	} else {
		invalidAction(action);
	}
}

LoadModelFromDisk::LoadModelFromDisk(ClockworkWorker* worker, std::shared_ptr<workerapi::LoadModelFromDisk> action) : 
		LoadModelFromDiskAction(worker->runtime, action), worker(worker) {
}

void LoadModelFromDisk::success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
	worker->controller->sendResult(result);
	delete this;
}

void LoadModelFromDisk::error(std::shared_ptr<workerapi::ErrorResult> result) {
	worker->controller->sendResult(result);
	delete this;
}


LoadWeights::LoadWeights(ClockworkWorker* worker, std::shared_ptr<workerapi::LoadWeights> action) :
		LoadWeightsAction(worker->runtime, action), worker(worker) {
}

void LoadWeights::success(std::shared_ptr<workerapi::LoadWeightsResult> result) {
	worker->controller->sendResult(result);
	delete this;
}

void LoadWeights::error(std::shared_ptr<workerapi::ErrorResult> result) {
	worker->controller->sendResult(result);
	delete this;
}

EvictWeights::EvictWeights(ClockworkWorker* worker, std::shared_ptr<workerapi::EvictWeights> action) :
		EvictWeightsAction(worker->runtime, action), worker(worker) {
}

void EvictWeights::success(std::shared_ptr<workerapi::EvictWeightsResult> result) {
	worker->controller->sendResult(result);
	delete this;
}

void EvictWeights::error(std::shared_ptr<workerapi::ErrorResult> result) {
	worker->controller->sendResult(result);
	delete this;
}

Infer::Infer(ClockworkWorker* worker, std::shared_ptr<workerapi::Infer> action) :
		InferAction(worker->runtime, action), worker(worker) {
}

void Infer::success(std::shared_ptr<workerapi::InferResult> result) {
	worker->controller->sendResult(result);
	delete this;
}

void Infer::error(std::shared_ptr<workerapi::ErrorResult> result) {
	worker->controller->sendResult(result);
	delete this;
}


}
