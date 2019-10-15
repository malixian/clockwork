#include <atomic>
#include "clockwork/network/worker_net.h"
#include "clockwork/network/client_net.h"
#include "clockwork/api/client_api.h"
#include "clockwork/api/worker_api.h"
#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <catch2/catch.hpp>
#include <nvml.h>
#include <iostream>
#include "clockwork/util.h"

using namespace clockwork;

std::string get_clockwork_dir() {
    int bufsize = 1024;
    char buf[bufsize];
    int len = readlink("/proc/self/exe", buf, bufsize);
	return dirname(dirname(buf));
}

std::string get_example_model(std::string name = "resnet18_tesla-m40_batchsize1") {
    return get_clockwork_dir() + "/resources/" + name + "/model";
}

class Controller : public workerapi::Controller, public clientapi::ClientAPI {
public:
	std::atomic_int model_id_seed;

	network::ControllerServer* client_facing_server;
	network::WorkerManager* worker_manager;
	std::vector<network::ControllerConnection*> workers;

	Controller(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs) : 
			model_id_seed(0) {
		client_facing_server = new network::ControllerServer(this, client_port);
		worker_manager = new network::WorkerManager();

		for (auto host_port_pair : worker_host_port_pairs) {
			network::ControllerConnection* connection = worker_manager->connect(host_port_pair.first, host_port_pair.second, this);
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

	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result) {
		std::cout << "Received result " << result->str() << std::endl;
	}

	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
		std::cout << "Controller uploadModel" << std::endl;
		clientapi::UploadModelResponse rsp;
		callback(rsp);
	}

	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
		std::cout << "Controller uploadModel" << std::endl;
		clientapi::InferenceResponse rsp;
		callback(rsp);
	}

	/** This is a 'backdoor' API function for ease of experimentation */
	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
		std::cout << "Controller uploadModel" << std::endl;
		clientapi::EvictResponse rsp;
		callback(rsp);
	}

	/** This is a 'backdoor' API function for ease of experimentation */
	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
		
		auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
		load_model->id = 0;
		load_model->model_id = model_id_seed++;
		load_model->model_path = get_example_model();
		load_model->earliest = 0;
		load_model->latest = util::now() + 10000000000L;

	// std::vector<std::shared_ptr<workerapi::Action>> actions;
	// actions = {load_model};

	// worker->sendActions(actions);


		clientapi::LoadModelFromRemoteDiskResponse rsp;
		callback(rsp);
	}
};

int main(int argc, char *argv[]) {
	std::cout << "Starting Clockwork Controller" << std::endl;

	int client_requests_listen_port = 12346;

	std::vector<std::pair<std::string, std::string>> worker_host_port_pairs = {
		{"127.0.0.1", "12345"}
	};

	Controller* controller = new Controller(client_requests_listen_port, worker_host_port_pairs);

	controller->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}