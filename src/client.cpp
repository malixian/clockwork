#include "clockwork/util.h"
#include "clockwork/client.h"
#include "clockwork/workload/workload.h"
#include "clockwork/workload/example.h"
#include <string>
#include <iostream>
#include "clockwork/workload/azure.h"
#include "clockwork/thread.h"

using namespace clockwork;

std::pair<std::string, std::string> split(std::string addr) {
	auto split = addr.find(":");
	std::string hostname = addr.substr(0, split);
	std::string port = addr.substr(split+1, addr.size());
	return {hostname, port};
}

void printUsage() {
	std::cerr << "Usage: client [address] [workload] "
			  << "[workload parameters (if required)]" << std::endl
			  << "Available workloads with parameters:" << std::endl
			  << "  example" << std::endl
			  << "  spam" << std::endl
			  << "          resnet50_v2 x 100, each with 100 closed loop" << std::endl
			  << "  single-spam" << std::endl
			  << "          resnet50_v2 x 1, with 1000 closed loop" << std::endl
			  << "  simple" << std::endl
			  << "  simple-slo-factor" << std::endl
			  << "          3 models with closed-loop concurrency of 1" << std::endl
			  << "          Updates each model's slo factor every 10 seconds" << std::endl
			  << "  simple-parametric num_models concurrency requests_per_model"
			  << std::endl
			  << "  azure" << std::endl
			  << "  azure_small" << std::endl;
}

int main(int argc, char *argv[])
{
	threading::initProcess();

	if (argc < 3) {
		printUsage();
		return 1;
	}
	std::string workload = argv[2];
	auto address = split(std::string(argv[1]));

	std::cout << "Running " << workload
	          << " on " << address.first 
	          << ":" << address.second << std::endl;


	bool verbose = false; // Log every request and response?
	bool summary = true;  // Log summary once per second?

	clockwork::Client *client = clockwork::Connect(
		address.first, address.second, 
		verbose, summary);

	workload::Engine* engine;
	if (workload == "example") 
		engine = workload::example(client);
	else if (workload == "spam") 
		engine = workload::spam(client);
	else if (workload == "single-spam") 
		engine = workload::single_spam(client);
	else if (workload == "simple")
		engine = workload::simple(client);
	else if (workload == "simple-slo-factor")
		engine = workload::simple_slo_factor(client);
	else if (workload == "simple-parametric")
		engine = workload::simple_parametric(client,
			std::stoul(argv[3]), std::stoul(argv[4]), std::stoul(argv[5]));
	else if (workload == "azure")
		engine = workload::azure(client);
	else if (workload == "azure_small")
		engine = workload::azure_small(client);
	else {
		std::cout << "Unknown workload " << workload << std::endl << std::endl;
		printUsage();
		return 1;
	}

	engine->Run(client);
}
