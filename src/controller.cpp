#include "clockwork/network/worker_net.h"
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

class Printer : public workerapi::Controller {
public:
	std::atomic_int results_count = 0;

	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result) {
		std::cout << "Received result " << result->str() << std::endl;
		results_count++;
	}

};

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: controller HOST PORT" << std::endl;
		return 1;
	}
	std::cout << "Starting Clockwork Controller" << std::endl;

	clockwork::network::WorkerClient* client = new clockwork::network::WorkerClient();

	Printer* controller = new Printer();
	auto worker = client->connect(argv[1], argv[2], controller);

	auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
	load_model->id = 0;
	load_model->model_id = 0;
	load_model->model_path = get_example_model();
	load_model->earliest = 0;
	load_model->latest = util::now() + 10000000000L;

	std::vector<std::shared_ptr<workerapi::Action>> actions;
	actions = {load_model};

	worker->sendActions(actions);

	while (controller->results_count.load() == 0);

	worker->close();
	client->shutdown(true);

	std::cout << "Clockwork Worker Exiting" << std::endl;
}