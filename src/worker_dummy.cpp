#include "clockwork/dummy/worker_dummy.h"
#include "clockwork/dummy/network/worker_dummy.h"
#include "clockwork/config.h"
#include "clockwork/thread.h"
#include <string>


int main(int argc, char *argv[]) {
	//threading::initProcess();
	//util::setCudaFlags();
	//util::printCudaVersion();
	//Don't need CUDA

	std::cout << "Starting Clockwork Worker" << std::endl;

	std::string config_file_path;
	std::string num_gpus = "2";//default

	for (int i = 1; i < argc; i++){
		
		if ((strcmp(argv[i], "--config") == 0))
			config_file_path = argv[++i];

		if ((strcmp(argv[i], "--num_gpus") == 0))
			num_gpus = argv[++i];
	}

	ClockworkWorkerConfig config(config_file_path);
	config.num_gpus = std::stoul(num_gpus,nullptr,0);

	clockwork::ClockworkDummyWorker* clockwork = new clockwork::ClockworkDummyWorker(config);
	//clockwork::ClockworkDummyWorker* clockwork = new clockwork::ClockworkDummyWorker();
	clockwork::network::worker::Server* server = new clockwork::network::worker::Server(clockwork);
	clockwork->setController(server);

	//threading::setDefaultPriority(); // Revert thread priority
	clockwork->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}
