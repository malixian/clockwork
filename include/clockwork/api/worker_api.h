#ifndef _CLOCKWORK_API_WORKER_API_H_
#define _CLOCKWORK_API_WORKER_API_H_

#include <functional>
#include <string>
#include "clockwork/api/api_common.h"

/**
This is the API for Clockwork Workers that are controlled by a centralized Clockwork scheduler.

This API is ONLY used by the clockwork runtime, which expects to receive specific actions to 
execute at specific times.
*/

const int actionSuccess = 0; // Action completed successfully
const int actionErrorUnknownModel = 1; // Action requested an unknown model
const int actionErrorCouldNotStartInTime = 2; // Action dropped because it could not be executed in time
const int actionErrorModelWeightsNotPresent = 3; // Infer or Evict action could not happen because no weights
const int actionErrorWeightsAlreadyLoaded = 4; // LoadWeightsAction failed because weights already loaded
const int actionErrorInvalidInput = 5; // Invalid input to an inference action
const int actionCancelled = 6; // Action cancelled for some other reason


namespace clockwork {
namespace workerapi {

/* Action types */
const int loadWeightsAction = 0;
const int evictWeightsAction = 1;
const int inferAction = 2;

/* The central scheduler sends actions to workers.  For now, use a single struct for actions
for simplicity, even though some fields won't make sense to use for some action types (especially
if we add new action types in future) */
struct Action {
	/* A unique ID for the action */
	int id;

	/* Action type */
	int action_type;

	/* ID of the model for this action */
	int model_id;

	/* Which GPU this action is referring to. */
	int gpu_id;

	/* The earliest time according to the worker's clock that the action may begin, in nanos.
	The action will not begin until at least this time has been reached. */
	uint64_t earliest;

	/* The latest time according to the worker's clock that the action may begin, in nanos. 
	It's recommended to just set earliest and latest to be some constant delta from each other.
	If the action cannot be started before this time, then it will be dropped. */
	uint64_t latest;

	/* This probably isn't used for anything, but is useful for logging telemetry.  The
	expected duration of this action as predicted by the scheduler */
	uint64_t expected_duration;

	/* Only used for exec actions -- the batch size to execute */
	int batch_size;

	/* Only used for exec actions -- the input data as a batch */
	int input_size;
	char* input;
};

/* The result of an action that is communicated back to the central scheduler */
struct Result {
	/* The unique ID of the action that this is a result of */
	int id;

	/* The status of the action (success, or action-specific failures, listed above )*/
	int status;

	/* The actual time according to the worker's clock that the action began, in nanos */
	uint64_t begin;

	/* The end time of the action according to the worker's clock, in nanos */
	uint64_t end;

	/* The duration of the action in nanos.  Note that this is NOT typically equal to 
	(end-begin).  Because of asynchrony with CUDA executions, this duration will be LESS
	than (end-begin), but more accurate for the purpose of estimating future action costs */
	uint64_t duration;

	/* Only used for exec actions -- the batch size that was executed */
	int batch_size;

	/* Only used for exec actions -- the output results as a batch */
	size_t output_size;
	char* output;
};

/* Model code for a specific batch size */
struct ModelInstance {
	/* Each batch size has different code and spec */
	int batch_size;
	size_t so_size;
	char* so;
	size_t spec_size;
	char* spec;
};

/* This is only currently used during setup/tear down phase of Clockwork */
struct UploadModelRequest {
	RequestHeader header;

	/* Weights are shared across batch sizes */
	size_t weights_size;
	void* weights_params;
	
	/* Code and params for different batch sizes */
	std::vector<ModelInstance> instances;
};

struct ModelInstanceStats {
	/* Each batch size has a different execution duration */
	int batch_size;
	uint64_t exec_duration;
};

/* This is only currently used during setup/tear down phase of Clockwork */
struct UploadModelResponse {
	ResponseHeader header;
	int model_id;

	/* Input and output sizes; assume a batch size of n multiples these by n. */
	size_t input_size;
	size_t output_size;

	/* Size of weights and time to load is independent of batch size */
	size_t weights_size_in_cache;
	uint64_t load_weights_duration;

	/* Execution duration varies based on batch size */
	std::vector<ModelInstanceStats> stats;
};

struct ActionSet {
	RequestHeader header;

	std::vector<Action> actions;
};

struct ResultSet {
	ResponseHeader header;

	std::vector<Result> results;
};

/** This is a 'backdoor' API function for ease of experimentation.  It returns an UploadModelResponse */
struct LoadModelFromRemoteDiskRequest {
	RequestHeader header;

	std::string remote_path;
};

class WorkerAPI {
public:

	virtual void uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback) = 0;

	virtual void addActions(ActionSet &actions, std::function<void(ResultSet&)> callback) = 0;

	/** This is a 'backdoor' API function for ease of experimentation.  It is similar to uploadModel only it
	loads a model from the worker's filesystem rather than using what was sent */
	virtual void loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(UploadModelResponse&)> callback) = 0;

};

}
}

#endif