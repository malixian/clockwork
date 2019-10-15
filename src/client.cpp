#include "clockwork/network/client.h"
#include "clockwork/api/client_api.h"
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

class ClosedLoopClient {
public:
	std::atomic_int request_id_seed;
	network::client::Connection* client;

	ClosedLoopClient(network::client::Connection* client) : client(client), request_id_seed(0) {
		loadModel();
	}

	void loadModel() {
		clientapi::LoadModelFromRemoteDiskRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.remote_path = get_example_model();

		std::cout << "<--  " << request.str() << std::endl;

		client->loadRemoteModel(request, [this] (clientapi::LoadModelFromRemoteDiskResponse &response) {
			std::cout << " --> " << response.str() << std::endl;
			if (response.header.status == clockworkSuccess) {
				this->infer(response.model_id, response.input_size);
			}
		});
	}

	void infer(int model_id, int input_size) {
		clientapi::InferenceRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.model_id = model_id;
		request.batch_size = 1;
		request.input_size = input_size;
		request.input = malloc(input_size);

		std::cout << "<--  " << request.str() << std::endl;

		client->infer(request, [this, model_id, input_size] (clientapi::InferenceResponse &response) {
			std::cout << " --> " << response.str() << std::endl;
			if (response.header.status == clockworkSuccess) {
				this->infer(model_id, input_size);
			}
		});		
	}

};

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: controller HOST PORT" << std::endl;
		return 1;
	}
	std::cout << "Starting Clockwork Client" << std::endl;

	// Manages client-side connections to clockwork, has internal network IO thread
	network::client::ConnectionManager* manager = new network::client::ConnectionManager();

	// Connect to clockwork
	network::client::Connection* clockwork_connection = manager->connect(argv[1], argv[2]);    

	// Simple closed-loop client
	ClosedLoopClient* closed_loop = new ClosedLoopClient(clockwork_connection);

	manager->join();

	std::cout << "Clockwork Client Exiting" << std::endl;
}