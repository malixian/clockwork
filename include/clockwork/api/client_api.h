#ifndef _CLOCKWORK_CLIENT_API_H_
#define _CLOCKWORK_CLIENT_API_H_

#include <functional>
#include <string>
#include "clockwork/api/api_common.h"

/**
This is the semi-public API between Clockwork front-end server and the Clockwork client library.

Clockwork clients should use the frontdoor API defined in client.h rather than this API, as this API has
more internal metadata than the frontdoor API.
*/

namespace clockwork {
namespace api {

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
	unsigned input_size;
	unsigned output_size;
};

struct InferenceRequest {
	RequestHeader header;
	int model_id;
	size_t input_size;
	void* input;
};

struct InferenceResponse {
	ResponseHeader header;
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
	unsigned input_size;
	unsigned output_size;
};


class ClientAPI {
public:

	/** The proper way of uploading a model will be to send it an ONNX file,
	where it will be compiled remotely.  For now we'll pre-compile clockwork
	models.  This is the synchronous version.  On error, will throw an exception. */
	void uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback);

	void infer(InferenceRequest &request, std::function<void(InferenceResponse&)> callback);

	/** This is a 'backdoor' API function for ease of experimentation */
	void loadModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback);

};

}
}

#endif