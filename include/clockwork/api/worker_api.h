#ifndef _CLOCKWORK_API_WORKER_API_H_
#define _CLOCKWORK_API_WORKER_API_H_

#include <functional>
#include <string>
#include <memory>
#include "clockwork/api/api_common.h"

/**
This is the API for Clockwork Workers that are controlled by a centralized Clockwork scheduler.

This API is ONLY used by the clockwork runtime, which expects to receive specific actions to 
execute at specific times.
*/

const int actionSuccess = 0; // Action completed successfully
const int actionCancelled = 1; // Action cancelled for some other reason
const int actionErrorRuntimeError = 2; // Action cancelled due to runtime error
const int actionErrorShuttingDown = 3; // Action cancelled due to clockwork shutting down

const int actionErrorUnknownModel = 10; // Action requested an unknown model
const int actionErrorCouldNotStartInTime = 11; // Action dropped because it could not be executed in time
const int actionErrorInvalidAction = 12; // An invalid action type was specified
const int actionErrorInvalidGPU = 13; // An invalid GPU device id was specified

const int actionErrorModelWeightsNotPresent = 20; // Infer or Evict action could not happen because no weights
const int actionErrorWeightsAlreadyLoaded = 21; // LoadWeightsAction failed because weights already loaded
const int actionErrorWeightsInUse = 22; // LoadWeightsAction failed because weights are being actively used (e.g. for transfer)
const int actionErrorWeightsChanged = 23; // Infer action failed because weights changed while executing

const int actionErrorInvalidInput = 30; // Invalid input to an inference action

const int actionErrorInvalidModelID = 40; // Invalid ID specified for load model
const int actionErrorInvalidModelPath = 41; // Invalid path specified for load model


namespace clockwork {
namespace workerapi {

/* Action types */
const int loadModelFromDiskAction = 0;
const int loadWeightsAction = 1;
const int inferAction = 2;
const int evictWeightsAction = 3;

class Action {
public:
	int id;
	int action_type;

	virtual std::string str() = 0;
};

class LoadModelFromDisk : public Action {
public:
	int model_id;
	std::string model_path;
	uint64_t earliest;
	uint64_t latest;

	virtual std::string str();
};

class LoadWeights : public Action {
public:
	uint64_t earliest;
	uint64_t latest;
	uint64_t expected_duration;

	int model_id;
	int gpu_id;
	
	virtual std::string str();
};

class EvictWeights : public Action {
public:
	uint64_t earliest;
	uint64_t latest;

	int model_id;
	int gpu_id;
	
	virtual std::string str();
};

class Infer : public Action {
public:
	uint64_t earliest;
	uint64_t latest;
	uint64_t expected_duration;

	int model_id;
	int gpu_id;
	int batch_size;
	int input_size;
	char* input;
	
	virtual std::string str();
};

class Result {
public:
	int id;
	int action_type;
	int status;
	
	virtual std::string str() = 0;
};

class ErrorResult : public Result {
public:
	std::string message;
	
	virtual std::string str();
};

class Timing {
public:
	uint64_t begin;
	uint64_t end;
	uint64_t duration; // For async tasks this is NOT end-begin
	
	virtual std::string str();
};

class LoadModelFromDiskResult : public Result, public Timing {
public:
	size_t input_size;
	size_t output_size;
	std::vector<unsigned> supported_batch_sizes;

	size_t weights_size_in_cache;

	// TODO: put some measurements like weight load time and exec time here
	
	virtual std::string str();
};

class LoadWeightsResult : public Result, public Timing {
public:
	
	virtual std::string str();
};

class EvictWeightsResult : public Result, public Timing {
public:
	
	virtual std::string str();
};

class InferResult : public Result {
public:
	Timing copy_input;
	Timing exec;
	Timing copy_output;
	int output_size;
	char* output;
	
	virtual std::string str();
};

// TODO: upload model action or RPC possibly

class Worker {
public:

	virtual void sendActions(std::vector<std::shared_ptr<Action>> &actions) = 0;

};

class Controller {
public:

	/* Although actions are communicated in batches, results are communicated immediately and individually */
	virtual void sendResult(std::shared_ptr<Result> result) = 0;

};

}
}

#endif