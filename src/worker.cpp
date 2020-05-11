#include "clockwork/worker.h"
#include "clockwork/network/worker.h"
#include "clockwork/runtime.h"
#include "clockwork/config.h"
#include "clockwork/worker.h"


int main(int argc, char *argv[]) {
	util::setCurrentThreadMaxPriority();
	util::setCudaFlags();
	util::printCudaVersion();

	std::cout << "Starting Clockwork Worker" << std::endl;

	std::string config_file_path;
	for (int i = 1; i < argc; i++){
		if ((strcmp(argv[i], "--config") == 0))
			config_file_path = argv[++i];
	}

	ClockworkWorkerConfig config(config_file_path);

	clockwork::ClockworkWorker* clockwork = new clockwork::ClockworkWorker(config);
	clockwork::network::worker::Server* server = new clockwork::network::worker::Server(clockwork);
	clockwork->controller = server;
	clockwork->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}
