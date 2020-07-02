#include "clockwork/worker_dummy.h"
#include "clockwork/network/worker_dummy.h"
#include "clockwork/config.h"
#include "clockwork/thread.h"


int main(int argc, char *argv[]) {
	threading::initProcess();
	util::setCudaFlags();
	util::printCudaVersion();

	std::cout << "Starting Clockwork Worker" << std::endl;

	std::string config_file_path;

	for (int i = 1; i < argc; i++){
		
		if ((strcmp(argv[i], "--config") == 0))
			config_file_path = argv[++i];
	}

	ClockworkWorkerConfig config(config_file_path);

	clockwork::ClockworkDummyWorker* clockwork = new clockwork::ClockworkDummyWorker(config);
	//clockwork::ClockworkDummyWorker* clockwork = new clockwork::ClockworkDummyWorker();
	clockwork::network::worker::Server* server = new clockwork::network::worker::Server(clockwork);
	clockwork->runtime->setController(server);
	clockwork->controller = server;

	threading::setDefaultPriority(); // Revert thread priority
	clockwork->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}
