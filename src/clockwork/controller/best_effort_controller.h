#include <atomic>
#include "clockwork/network/controller.h"
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
#include "controller.h"

using namespace clockwork;

// Simple controller implementation that forwards all client requests to workers
class BestEffortControllerImpl : public Controller {
public:
	BestEffortControllerImpl(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs);

	virtual void sendResult(std::shared_ptr<workerapi::Result> result);
	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback);
	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback);
	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback);
	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback);
};


// Scheduler will make decisions this far ahead of time
#define SCHEDULING_BUFFER 1000000UL

// Buffer time for copying inputs and outputs for infer action + network
#define IO_OVERHEAD 2000000UL


// Loading model from remote disk on the worker is a setup-stage task
enum Mode { normal, pre_loadmodel, loadmodel };

class WorkerState {
public:


	uint64_t pci_busy_until = 0;
	std::array<uint64_t, 2> gpu_busy_until = 0;

	tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> results;

	void schedule(uint64_t now, GlobalState* globalState) {


	}

}