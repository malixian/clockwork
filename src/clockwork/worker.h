#ifndef _CLOCKWORK_WORKER_H_
#define _CLOCKWORK_WORKER_H_

#include "clockwork/action.h"
#include "clockwork/api/worker_api.h"

/*
This file ties together the worker API (defined in api/worker_api.h) with model actions (defined in action.h)
using a clockwork scheduling framework (defined in runtime.h).
*/

namespace clockwork {

class ClockworkWorker : public workerapi::Worker {
public:
	ClockworkRuntime* runtime;
	workerapi::Controller* controller;

	// TODO: actually instantiate the clockwork runtime properly and set the controller
	ClockworkWorker();
	~ClockworkWorker();

	void shutdown(bool await_completion);
	void join();

	void sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions);

private:
	void invalidAction(std::shared_ptr<workerapi::Action> action);
	void loadModel(std::shared_ptr<workerapi::Action> action);
	void loadWeights(std::shared_ptr<workerapi::Action> action);
	void evictWeights(std::shared_ptr<workerapi::Action> action);
	void infer(std::shared_ptr<workerapi::Action> action);
	void clearCache(std::shared_ptr<workerapi::Action> action);
	void getWorkerState(std::shared_ptr<workerapi::Action> action);
};

class LoadModelFromDisk : public LoadModelFromDiskAction {
public:
	ClockworkWorker* worker;

	LoadModelFromDisk(ClockworkWorker* worker, std::shared_ptr<workerapi::LoadModelFromDisk> action);

	void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result);
	void error(std::shared_ptr<workerapi::ErrorResult> result);
};

class LoadWeights : public LoadWeightsAction {
public:
	ClockworkWorker* worker;

	LoadWeights(ClockworkWorker* worker, std::shared_ptr<workerapi::LoadWeights> action);

	void success(std::shared_ptr<workerapi::LoadWeightsResult> result);
	void error(std::shared_ptr<workerapi::ErrorResult> result);
};

class EvictWeights : public EvictWeightsAction {
public:
	ClockworkWorker* worker;

	EvictWeights(ClockworkWorker* worker, std::shared_ptr<workerapi::EvictWeights> action);

	void success(std::shared_ptr<workerapi::EvictWeightsResult> result);
	void error(std::shared_ptr<workerapi::ErrorResult> result);
};

class Infer : public InferAction {
public:
	ClockworkWorker* worker;

	Infer(ClockworkWorker* worker, std::shared_ptr<workerapi::Infer> action);

	void success(std::shared_ptr<workerapi::InferResult> result);
	void error(std::shared_ptr<workerapi::ErrorResult> result);
};



}

#endif
