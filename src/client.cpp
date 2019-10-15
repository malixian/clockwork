#include "clockwork/network/client_net.h"
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
	network::ControllerClient* client;

	ClosedLoopClient(network::ControllerClient* client) : client(client) {
		loadModel();
	}

	void loadModel() {
		clientapi::LoadModelFromRemoteDiskRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = 0;
		request.remote_path = get_example_model();

		client->loadRemoteModel(request, [this] (clientapi::LoadModelFromRemoteDiskResponse &response) {
			std::cout << "Model loaded" << std::endl;
			std::cout << response.header.status << ": " << response.header.message << std::endl;
			std::cout << response.model_id << " input=" << response.input_size << " output=" << response.output_size << std::endl;
			this->infer();
		});
	}

	void infer() {
		std::cout << "inferring" << std::endl;
	}

};

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: controller HOST PORT" << std::endl;
		return 1;
	}
	std::cout << "Starting Clockwork Client" << std::endl;

	asio::io_service io_service;

	network::ControllerClient* network_client = new network::ControllerClient(io_service);

	ClosedLoopClient* closed_loop = nullptr;
    network_client->set_ready_cb([network_client, &closed_loop](){ 
    	closed_loop = new ClosedLoopClient(network_client);
    });
    network_client->connect(argv[1], argv[2]);

	try {
	    io_service.run();
	} catch (std::exception& e) {
	    std::cerr << e.what() << std::endl;
	} catch (const char * m) {
	    std::cerr << m << std::endl;
	}

	std::cout << "Clockwork Client Exiting" << std::endl;
}