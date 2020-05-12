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

int main(int argc, char *argv[])
{
	threading::initProcess();

	if (argc != 3) {
		std::cerr << "Usage: client [address] [workload]" << std::endl;
		std::cout << "Available workloads:" << std::endl;
		std::cout << "  simple" << std::endl;
		std::cout << "  example" << std::endl;
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
	else if (workload == "simple")
		engine = workload::simple(client);
	else if (workload == "azure")
		engine = workload::azure(client);
	else {
		std::cout << "Unknown workload " << workload << std::endl;
		return 1;
	}

	engine->Run(client);
}
