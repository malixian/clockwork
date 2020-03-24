#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "clockwork/network/controller.h"
#include "clockwork/api/worker_api.h"

using namespace clockwork;

class Controller : public workerapi::Controller, public clientapi::ClientAPI {
public:
	network::controller::Server* client_facing_server;
	network::controller::WorkerManager* worker_manager;
	std::vector<network::controller::WorkerConnection*> workers;
	int worker = 0;
	Controller(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs) {
		client_facing_server = new network::controller::Server(this, client_port);
		worker_manager = new network::controller::WorkerManager();

		for (auto host_port_pair : worker_host_port_pairs) {
			network::controller::WorkerConnection* connection = worker_manager->connect(host_port_pair.first, host_port_pair.second, this);
			workers.push_back(connection);
		}
	}

	void shutdown(bool awaitShutdown = false) {
		// TODO
		if (awaitShutdown) {
			join();
		}
	}

	void join() {
		// TODO
		worker_manager->join();
	}

	// workerapi -- results received from workers call these functions
	virtual void sendResult(std::shared_ptr<workerapi::Result> result) = 0;

	// clientapi -- requests from clients call these functions
	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) = 0;
	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) = 0;
	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) = 0;
	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) = 0;
};

#endif 

