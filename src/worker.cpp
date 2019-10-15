#include "clockwork/worker.h"
#include "clockwork/network/worker.h"


int main(int argc, char *argv[]) {
	std::cout << "Starting Clockwork Worker" << std::endl;

	clockwork::ClockworkWorker* clockwork = new clockwork::ClockworkWorker();
	clockwork::network::worker::Server* server = new clockwork::network::worker::Server(clockwork);
	clockwork->controller = server;
	clockwork->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}