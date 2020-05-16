#include "clockwork/util.h"
#include "clockwork/client.h"
#include "clockwork/workload/workload.h"
#include "clockwork/workload/example.h"
#include <string>
#include <iostream>
#include "clockwork/workload/azure.h"
#include "clockwork/workload/slo.h"
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
			  << "\t example" << std::endl
			  << "\t spam" << std::endl
			  << "\t\t resnet50_v2 x 100, each with 100 closed loop" << std::endl
			  << "\t single-spam" << std::endl
			  << "\t\t resnet50_v2 x 1, with 1000 closed loop" << std::endl
			  << "\t simple" << std::endl
			  << "\t simple-slo-factor" << std::endl
			  << "\t\t 3 models with closed-loop concurrency of 1" << std::endl
			  << "\t\t Updates each model's slo factor every 10 seconds" << std::endl
			  << "\t simple-parametric num_models concurrency requests_per_model" << std::endl
			  << "\t poisson-open-loop num_models rate " << std::endl
			  << "\t\t Rate should be provided in requests/second" << std::endl
			  << "\t\t Rate is split across all models" << std::endl
			  << "\t slo-exp-1 model copies dist rate slo-start slo-end slo-factor slo-op period" << std::endl
			  << "\t\t Workload parameters:" << std::endl
			  << "\t\t\t model: model name (e.g., \"resnet50_v2\")" << std::endl
			  << "\t\t\t copies: number of model instances" << std::endl
			  << "\t\t\t dist: arrival distribution (\"poisson\"/\"fixed-rate\")" << std::endl
			  << "\t\t\t rate: arrival rate (in requests/second)" << std::endl
			  << "\t\t\t slo-start: starting slo value (in ms)" << std::endl
			  << "\t\t\t slo-end: ending slo value (in ms)" << std::endl
			  << "\t\t\t slo-factor: factor by which the slo should change" << std::endl
			  << "\t\t\t slo-op: operator (\"add\"/\"mul\") for incrementing slo" << std::endl
			  << "\t\t\t period: number of seconds before changing slo" << std::endl
			  << "\t\t Examples:" << std::endl
			  << "\t\t\t client volta04:12346 slo-exp-1 resnet50_v2 4 poisson 100 2 32 2 mul 7" << std::endl
			  << "\t\t\t\t (increases slo every 7s as follows: 2 4 8 16 32)" << std::endl
			  << "\t\t\t client volta04:12346 slo-exp-1 resnet50_v2 4 poisson 100 10 100 10 add 3" << std::endl
			  << "\t\t\t\t (increases slo every 3s as follows: 10 20 30 ... 100)" << std::endl
			  << "\t\t In each case, an open loop client is used" << std::endl
			  << "\tazure" << std::endl
			  << "\tazure_small" << std::endl;
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
	else if (workload == "slo-exp-1")
		engine = workload::slo_experiment_1(
			client,
			std::string(argv[3]), 	// model name
			std::stoul(argv[4]),	// num of copies
			std::string(argv[5]),	// arrival distribution
			std::stod(argv[6]),		// arrival rate (requests/second)
			std::stoull(argv[7]),	// starting slo (in ms)
			std::stoull(argv[8]),	// ending slo (in ms)
			std::stoull(argv[9]),	// slo increase factor
			std::string(argv[10]),	// slo increase operator
			std::stoull(argv[11]));	// slo step duration (in seconds)
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
