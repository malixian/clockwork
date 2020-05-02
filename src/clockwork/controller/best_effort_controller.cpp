#include <atomic>
#include "clockwork/network/controller.h"
#include "direct_controller.h"
#include "clockwork/api/client_api.h"
#include "clockwork/api/worker_api.h"
#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <nvml.h>
#include <iostream>
#include "clockwork/util.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <dmlc/logging.h>

using namespace clockwork;

BestEffortControllerImpl::BestEffortControllerImpl(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs) :
	Controller(client_port, worker_host_port_pairs), model_id_seed(0), action_id_seed(0) {}

// workerapi::Controller::sendResult
void BestEffortControllerImpl::sendResult(std::shared_ptr<workerapi::Result> result) {
}

void DirectControllerImpl::infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
}

/** This is a 'backdoor' API function for ease of experimentation */
void DirectControllerImpl::loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
	
}

void DirectControllerImpl::uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
	CHECK(false) << "DirectControllerImpl::uploadModel is called but not implemented";
}

/** This is a 'backdoor' API function for ease of experimentation */
void DirectControllerImpl::evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
	CHECK(false) << "DirectControllerImpl::evict is called but not implemented";
}

