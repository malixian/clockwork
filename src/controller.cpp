#include "clockwork/network/controller.h"
#include "clockwork/controller/direct_controller.h"
#include "clockwork/controller/closed_loop_controller.h"


using namespace clockwork;

int main(int argc, char *argv[]) {
	std::cout << "Starting Clockwork Controller" << std::endl;

	if ( argc < 3) {
		std::cerr << "USAGE ./controller [CLOSED_LOOP/DIRECT] MAX_BATH_SIZE" << std::endl;
		return 1;
	}
	
	std::string controller_type = argv[1];

	int batch_size = atoi(argv[2]);

	int client_requests_listen_port = 12346;

	std::vector<std::pair<std::string, std::string>> worker_host_port_pairs = {
		{"127.0.0.1", "12345"}
	};

	if ( controller_type == "CLOSED_LOOP"){
		ClosedLoopControllerImpl* controller = new ClosedLoopControllerImpl(client_requests_listen_port, worker_host_port_pairs, batch_size);
		controller->join();
	} else if (controller_type == "DIRECT") {
		DirectControllerImpl* controller = new DirectControllerImpl(client_requests_listen_port, worker_host_port_pairs);
		controller->join();

	}

	std::cout << "Clockwork Worker Exiting" << std::endl;
}
