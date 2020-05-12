#include "clockwork/network/controller.h"
#include "clockwork/controller/direct_controller.h"
#include "clockwork/controller/closed_loop_controller.h"
#include "clockwork/controller/stress_test_controller.h"
#include "clockwork/controller/infer_only_scheduler.h"
#include "clockwork/telemetry/controller_request_logger.h"
#include "clockwork/thread.h"


using namespace clockwork;

int main(int argc, char *argv[]) {
	threading::initProcess();	

	std::cout << "Starting Clockwork Controller" << std::endl;

	if ( argc < 4) {
		std::cerr << "USAGE ./controller [CLOSED_LOOP/DIRECT/STRESS/ECHO] MAX_BATCH_SIZE worker1:port1 worker2:port2 ..." << std::endl;
		return 1;
	}
	
	std::string controller_type = argv[1];

	int batch_size = atoi(argv[2]);

	int client_requests_listen_port = 12346;


	std::vector<std::pair<std::string, std::string>> worker_host_port_pairs;
	for (int i = 3; i < argc; i++) {
		std::string addr = std::string(argv[i]);
		auto split = addr.find(":");
		std::string hostname = addr.substr(0, split);
		std::string port = addr.substr(split+1, addr.size());
		worker_host_port_pairs.push_back({hostname, port});
	}

	if ( controller_type == "CLOSED_LOOP"){
		ClosedLoopControllerImpl* controller = new ClosedLoopControllerImpl(client_requests_listen_port, worker_host_port_pairs, batch_size);
		controller->join();
	} else if (controller_type == "DIRECT") {
		DirectControllerImpl* controller = new DirectControllerImpl(client_requests_listen_port, worker_host_port_pairs);
		controller->join();
	} else if (controller_type == "STRESS") {
		StressTestController* controller = new StressTestController(client_requests_listen_port, worker_host_port_pairs);
		controller->join();
	} else if (controller_type == "INFER") {
		Scheduler* scheduler = new InferOnlyScheduler();
		controller::ControllerWithStartupPhase* controller = new controller::ControllerWithStartupPhase(
			client_requests_listen_port,
			worker_host_port_pairs,
			10000000000UL, // 10s load stage timeout
			new controller::ControllerStartup(), // in future the startup type might be customizable
			scheduler,
			RequestTelemetryPrinter::async_request_printer(10000000000UL) // print request summary every 10s
		);
		controller->join();
	} else if (controller_type == "ECHO") {
		Scheduler* scheduler = new EchoScheduler();
		controller::ControllerWithStartupPhase* controller = new controller::ControllerWithStartupPhase(
			client_requests_listen_port,
			worker_host_port_pairs,
			10000000000UL, // 10s load stage timeout
			new controller::ControllerStartup(), // in future the startup type might be customizable
			scheduler,
			RequestTelemetryPrinter::async_request_printer(10000000000UL) // print request summary every 10s
		);
		controller->join();
	}

	std::cout << "Clockwork Worker Exiting" << std::endl;
}
