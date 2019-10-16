#include "clockwork/network/client.h"
#include "clockwork/api/client_api.h"
#include "clockwork/client.h"
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

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: controller HOST PORT" << std::endl;
		return 1;
	}
	std::cout << "Starting Clockwork Client" << std::endl;

	clockwork::Client* client = clockwork::Connect(argv[1], argv[2]);

	clockwork::Model* model = client->load_remote_model(get_example_model());

	while (true) {
		std::vector<uint8_t> input(model->input_size());
		model->infer(input);		
	}

	std::cout << "Clockwork Client Exiting" << std::endl;
}