#ifndef _CLOCKWORK_API_WORKER_API_H_
#define _CLOCKWORK_API_WORKER_API_H_

#include <functional>
#include <string>
#include "clockwork/api/api_common.h"

/**
This is the API for Clockwork Workers that are controlled by a centralized Clockwork scheduler
*/

namespace clockwork {
namespace api {
namespace worker {

struct UploadModelRequest {
	RequestHeader header;
	size_t so_size;
	void* so;
	size_t weights_size;
	void* weights_params;
	size_t clockwork_spec_size;
	void* clockwork_spec;
};

struct UploadModelResponse {
	ResponseHeader header;
	int model_id;
	size_t input_size;
	size_t output_size;
	size_t size_in_cache;
};

struct ActionRequest {
	int gpu_id;
	uint64_t action_id;
	std::vector<uint64_t> happens_after;
	uint64_t priority;
};

struct ActionResponse {
	int gpu_id;
	uint64_t duration;
};

struct LoadWeightsAction {
	RequestHeader header;
	ActionRequest action;
	int model_id;
};

struct LoadWeightsResponse {
	ResponseHeader header;
	ActionResponse action;
	int model_id;
};

struct UnloadWeightsAction {
	RequestHeader header;
	ActionRequest action;
	int model_id;
};

struct UnloadWeightsResponse {
	ResponseHeader header;
	ActionResponse action;
	int model_id;
};

struct ExecuteAction {
	RequestHeader header;
	ActionRequest action;
	int model_id;
	size_t input_size;
	void* input;
};

struct ExecuteResponse {
	RequestHeader header;
	ActionResponse action;
	int model_id;
	size_t output_size;
	void* output;
};

/** This is a 'backdoor' API function for ease of experimentation */
struct LoadModelFromRemoteDiskRequest {
	RequestHeader header;
	std::string remote_path;
};

/** This is a 'backdoor' API function for ease of experimentation */
struct LoadModelFromRemoteDiskResponse {
	ResponseHeader header;
	int model_id;
	size_t input_size;
	size_t output_size;
	size_t size_in_cache;
};

class WorkerAPI {
public:

	/** The proper way of uploading a model will be to send it an ONNX file,
	where it will be compiled remotely.  For now we'll pre-compile clockwork
	models.  This is the synchronous version.  On error, will throw an exception. */
	void uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback);

	void loadWeights(LoadWeightsAction &request, std::function<void(LoadWeightsResponse&)> callback);

	void unloadWeights(UnloadWeightsAction &request, std::function<void(UnloadWeightsResponse&)> callback);

	void execute(ExecuteAction &request, std::function<void(ExecuteResponse&)> callback);

	/** This is a 'backdoor' API function for ease of experimentation */
	void loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback);

};

}
}
}

#endif