#ifndef _CLOCKWORK_WORKER_H_
#define _CLOCKWORK_WORKER_H_

#include "clockwork/api/worker_api.h"

using namespace clockwork::workerapi;

namespace clockwork {

class Worker : public WorkerAPI {
public:

	void uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback) {
		UploadModelResponse response;
		response.header.user_request_id = request.header.user_request_id;
		response.header.status = clockworkError;
		response.header.message = "uploadModel unimplemented";
		callback(response);
	}

	void loadWeights(LoadWeightsAction &request, std::function<void(LoadWeightsResponse&)> callback) {

	}

	virtual void unloadWeights(UnloadWeightsAction &request, std::function<void(UnloadWeightsResponse&)> callback) = 0;

	virtual void execute(ExecuteAction &request, std::function<void(ExecuteResponse&)> callback) = 0;

	/** This is a 'backdoor' API function for ease of experimentation */
	virtual void loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback) = 0;

};

}

#endif