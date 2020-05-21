#include "clockwork/network/controller.h"
#include "clockwork/api/worker_api.h"
#include "controller.h"

#ifndef GPUS_PER_WORKER
#define GPUS_PER_WORKER 2 
#endif

using namespace clockwork;

// Multi Machine, Multi GPU FIFO Controller
class FifoControllerImpl : public Controller {
public:
	std::atomic_int action_id_seed;
	std::atomic_int model_id_seed;
	std::mutex actions_mutex;
	std::unordered_map<int, std::function<void(std::shared_ptr<workerapi::Result>)>> action_callbacks;
	std::unordered_map<std::pair<int, unsigned>, uint64_t, util::hash_pair> weights_available_at; 
	std::unordered_map<std::tuple<int, unsigned, unsigned>, uint64_t, util::hash_tuple > new_weights_available_at; // model_id, worker_id, gpu_id 
	unsigned current_worker;
	std::map<unsigned, unsigned> current_worker_gpu;

	std::map<int, std::atomic_int> load_model_tracking;

	FifoControllerImpl(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs);

	void save_callback(int id, std::function<void(std::shared_ptr<workerapi::Result>)> callback);

	std::function<void(std::shared_ptr<workerapi::Result>)> get_callback(int id);

	void closedLoopController();

	void load_example_model();
	void load_weights_example_model(int gpu_id, int model_id);

	virtual void sendResult(std::shared_ptr<workerapi::Result> result);
	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback);
	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback);
	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback);
	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) ;
};

