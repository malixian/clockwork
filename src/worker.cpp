#include "clockwork/worker.h"
#include "clockwork/network/worker_net.h"


int main(int argc, char *argv[]) {
	std::cout << "Starting Clockwork Worker" << std::endl;

	clockwork::ClockworkWorker* clockwork = new clockwork::ClockworkWorker();
	clockwork::network::WorkerServer* server = new clockwork::network::WorkerServer(clockwork);
	clockwork->controller = server;
	clockwork->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}