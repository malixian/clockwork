#include "clockwork/worker.h"
#include <algorithm>

namespace clockwork {

void set_and_log_actionTelemetry(
		std::shared_ptr<ActionTelemetry> telemetry, ClockworkRuntime* runtime,
		int telemetry_type, int action_id, int action_type, int status,
		clockwork::time_point timestamp){
	telemetry->telemetry_type = telemetry_type;
	telemetry->action_id = action_id;
	telemetry->action_type = action_type;
	telemetry->status = status;
	telemetry->timestamp = timestamp;

	runtime->action_telemetry_logger->log(telemetry);
}

// TODO: actually instantiate the clockwork runtime properly and set the controller
ClockworkWorker::ClockworkWorker() : 
		runtime(new ClockworkRuntime()) {
}

ClockworkWorker::ClockworkWorker(ClockworkWorkerConfig &config) :
		runtime(new ClockworkRuntime(config)) {
}
ClockworkWorker::~ClockworkWorker() {
	this->shutdown(false);
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
		switch (action->action_type) {
			case workerapi::loadModelFromDiskAction: loadModel(action); break;
			case workerapi::loadWeightsAction: loadWeights(action); break;
			case workerapi::inferAction: infer(action); break;
			case workerapi::evictWeightsAction: evictWeights(action); break;
			case workerapi::clearCacheAction: clearCache(action); break;
			case workerapi::getWorkerStateAction: getWorkerState(action); break;
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

// Need to be careful of timestamp = 0 and timestamp = UINT64_MAX which occur often
// and clock_delta can be positive or negative
uint64_t adjust_timestamp(uint64_t timestamp, int64_t clock_delta) {
	if (clock_delta >= 0) return std::max(timestamp, timestamp + clock_delta);
	else return std::min(timestamp, timestamp + clock_delta);
}

void ClockworkWorker::loadModel(std::shared_ptr<workerapi::Action> action) {
	auto load_model = std::static_pointer_cast<workerapi::LoadModelFromDisk>(action);
	if (load_model != nullptr) {
		// It is a hack to do this here, but easiest / safest place to do it for now
		load_model->earliest = adjust_timestamp(load_model->earliest, load_model->clock_delta);
		load_model->latest = adjust_timestamp(load_model->latest, load_model->clock_delta);

		LoadModelFromDisk* action = new LoadModelFromDisk(this, load_model);
		action->submit();
	} else {
		invalidAction(action);
	}
}

void ClockworkWorker::loadWeights(std::shared_ptr<workerapi::Action> action) {
	auto load_weights = std::static_pointer_cast<workerapi::LoadWeights>(action);
	if (load_weights != nullptr) {
		// It is a hack to do this here, but easiest / safest place to do it for now
		load_weights->earliest = adjust_timestamp(load_weights->earliest, load_weights->clock_delta);
		load_weights->latest = adjust_timestamp(load_weights->latest, load_weights->clock_delta);

		LoadWeights* action = new LoadWeights(this, load_weights);
		action->submit();
	} else {
		invalidAction(action);
	}
}

void ClockworkWorker::evictWeights(std::shared_ptr<workerapi::Action> action) {
	auto evict_weights = std::static_pointer_cast<workerapi::EvictWeights>(action);
	if (evict_weights != nullptr) {
		// It is a hack to do this here, but easiest / safest place to do it for now
		evict_weights->earliest = adjust_timestamp(evict_weights->earliest, evict_weights->clock_delta);
		evict_weights->latest = adjust_timestamp(evict_weights->latest, evict_weights->clock_delta);

		EvictWeights* action = new EvictWeights(this, evict_weights);
		action->submit();
	} else {
		invalidAction(action);
	}
}

void ClockworkWorker::infer(std::shared_ptr<workerapi::Action> action) {
	auto infer = std::static_pointer_cast<workerapi::Infer>(action);
	if (infer != nullptr) {
		// It is a hack to do this here, but easiest / safest place to do it for now
		infer->earliest = adjust_timestamp(infer->earliest, infer->clock_delta);
		infer->latest = adjust_timestamp(infer->latest, infer->clock_delta);

		Infer* action = new Infer(this, infer);
		action->submit();
	} else {
		invalidAction(action);
	}
}

void ClockworkWorker::clearCache(std::shared_ptr<workerapi::Action> action) {
	auto clear_cache = std::static_pointer_cast<workerapi::ClearCache>(action);
	if (clear_cache != nullptr) {
		for (unsigned i = 0; i < runtime->num_gpus; i++) {
			runtime->manager->weights_caches[i]->clear();
		}
		auto result = std::make_shared<workerapi::ClearCacheResult>();
		result->id = action->id;
		result->action_type = workerapi::clearCacheAction;
		result->status = actionSuccess; // TODO What about error handling?
		controller->sendResult(result);
	} else {
		invalidAction(action);
	}
}

void ClockworkWorker::getWorkerState(std::shared_ptr<workerapi::Action> action) {
	auto get_worker_state = std::static_pointer_cast<workerapi::GetWorkerState>(action);
	if (get_worker_state != nullptr) {
		auto result = std::make_shared<workerapi::GetWorkerStateResult>();
		result->id = action->id;
		result->action_type = workerapi::getWorkerStateAction;
		runtime->manager->get_worker_memory_info(result->worker);
		result->status = actionSuccess; // TODO What about error handling?
		controller->sendResult(result);
	} else {
		invalidAction(action);
	}
}

LoadModelFromDisk::LoadModelFromDisk(ClockworkWorker* worker, std::shared_ptr<workerapi::LoadModelFromDisk> action) : 
		LoadModelFromDiskAction(worker->runtime, action), worker(worker),
		action_telemetry(std::make_shared<ActionTelemetry>()),
		response_telemetry(std::make_shared<ActionTelemetry>()) {
	// set_and_log_actionTelemetry(action_telemetry, runtime, 0, action->id, workerapi::loadModelFromDiskAction, 0, util::hrt());
}

void LoadModelFromDisk::success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
	// set_and_log_actionTelemetry(response_telemetry, runtime, 1, result->id, workerapi::loadModelFromDiskAction, actionSuccess, util::hrt());

	// It is a hack to do this here, but easiest / safest place to do it for now
	result->begin = adjust_timestamp(result->begin, -action->clock_delta);
	result->end = adjust_timestamp(result->end, -action->clock_delta);
	result->action_received = adjust_timestamp(action->received, -action->clock_delta);
	result->result_sent = util::now() - action->clock_delta;

	worker->controller->sendResult(result);
	delete this;
}

void LoadModelFromDisk::error(std::shared_ptr<workerapi::ErrorResult> result) {
	// set_and_log_actionTelemetry(response_telemetry, runtime, 1, result->id, workerapi::loadModelFromDiskAction, result->status, util::hrt());
	result->action_received = adjust_timestamp(action->received, -action->clock_delta);
	result->result_sent = util::now() - action->clock_delta;
	worker->controller->sendResult(result);
	delete this;
}


LoadWeights::LoadWeights(ClockworkWorker* worker, std::shared_ptr<workerapi::LoadWeights> action) :
		LoadWeightsAction(worker->runtime, action), worker(worker),
		action_telemetry(std::make_shared<ActionTelemetry>()),
		response_telemetry(std::make_shared<ActionTelemetry>()) {
	// set_and_log_actionTelemetry(action_telemetry, runtime, 0, action->id, workerapi::loadWeightsAction, 0, util::hrt());
}

void LoadWeights::success(std::shared_ptr<workerapi::LoadWeightsResult> result) {
	// set_and_log_actionTelemetry(response_telemetry, runtime, 1, result->id, workerapi::loadWeightsAction, actionSuccess, util::hrt());
	
	// It is a hack to do this here, but easiest / safest place to do it for now
	result->begin = adjust_timestamp(result->begin, -action->clock_delta);
	result->end = adjust_timestamp(result->end, -action->clock_delta);
	result->action_received = adjust_timestamp(action->received, -action->clock_delta);
	result->result_sent = util::now() - action->clock_delta;
	
	worker->controller->sendResult(result);
	delete this;
}

void LoadWeights::error(std::shared_ptr<workerapi::ErrorResult> result) {
	// set_and_log_actionTelemetry(response_telemetry, runtime, 1, result->id, workerapi::loadWeightsAction, result->status, util::hrt());
	result->action_received = adjust_timestamp(action->received, -action->clock_delta);
	result->result_sent = util::now() - action->clock_delta;
	worker->controller->sendResult(result);
	delete this;
}

EvictWeights::EvictWeights(ClockworkWorker* worker, std::shared_ptr<workerapi::EvictWeights> action) :
		EvictWeightsAction(worker->runtime, action), worker(worker),
		action_telemetry(std::make_shared<ActionTelemetry>()),
		response_telemetry(std::make_shared<ActionTelemetry>()) {
	// set_and_log_actionTelemetry(action_telemetry, runtime, 0, action->id, workerapi::evictWeightsAction, 0, util::hrt());
}

void EvictWeights::success(std::shared_ptr<workerapi::EvictWeightsResult> result) {
	// set_and_log_actionTelemetry(response_telemetry, runtime, 1, result->id, workerapi::evictWeightsAction, actionSuccess, util::hrt());
	
	// It is a hack to do this here, but easiest / safest place to do it for now
	result->begin = adjust_timestamp(result->begin, -action->clock_delta);
	result->end = adjust_timestamp(result->end, -action->clock_delta);
	result->action_received = adjust_timestamp(action->received, -action->clock_delta);
	result->result_sent = util::now() - action->clock_delta;
	
	worker->controller->sendResult(result);
	delete this;
}

void EvictWeights::error(std::shared_ptr<workerapi::ErrorResult> result) {
	// set_and_log_actionTelemetry(response_telemetry, runtime, 1, result->id, workerapi::evictWeightsAction, result->status, util::hrt());
	result->action_received = adjust_timestamp(action->received, -action->clock_delta);
	result->result_sent = util::now() - action->clock_delta;
	worker->controller->sendResult(result);
	delete this;
}

Infer::Infer(ClockworkWorker* worker, std::shared_ptr<workerapi::Infer> action) :
		InferAction(worker->runtime, action), worker(worker),
		action_telemetry(std::make_shared<ActionTelemetry>()),
		response_telemetry(std::make_shared<ActionTelemetry>())   {
	// set_and_log_actionTelemetry(action_telemetry, runtime, 0, action->id, workerapi::inferAction, 0, util::hrt());
}

void Infer::success(std::shared_ptr<workerapi::InferResult> result) {
	// set_and_log_actionTelemetry(response_telemetry, runtime, 1, result->id, workerapi::inferAction, actionSuccess, util::hrt());
	
	// It is a hack to do this here, but easiest / safest place to do it for now
	result->copy_input.begin = adjust_timestamp(result->copy_input.begin, -action->clock_delta);
	result->copy_input.end = adjust_timestamp(result->copy_input.end, -action->clock_delta);
	result->exec.begin = adjust_timestamp(result->exec.begin, -action->clock_delta);
	result->exec.end = adjust_timestamp(result->exec.end, -action->clock_delta);
	result->copy_output.begin = adjust_timestamp(result->copy_output.begin, -action->clock_delta);
	result->copy_output.end = adjust_timestamp(result->copy_output.end, -action->clock_delta);
	result->action_received = adjust_timestamp(action->received, -action->clock_delta);
	result->result_sent = util::now() - action->clock_delta;
	
	worker->controller->sendResult(result);
	delete this;
}

void Infer::error(std::shared_ptr<workerapi::ErrorResult> result) {
	// set_and_log_actionTelemetry(response_telemetry, runtime, 1, result->id, workerapi::inferAction, result->status, util::hrt());
	result->action_received = adjust_timestamp(action->received, -action->clock_delta);
	result->result_sent = util::now() - action->clock_delta;
	worker->controller->sendResult(result);
	delete this;
}


}
