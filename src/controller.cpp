#include "clockwork/network/controller.h"
#include "clockwork/controller/direct_controller.h"
#include "clockwork/controller/closed_loop_controller.h"
#include "clockwork/controller/stress_test_controller.h"
#include "clockwork/controller/infer_only_scheduler.h"
#include "clockwork/telemetry/controller_request_logger.h"
#include <csignal>
#include "clockwork/thread.h"


using namespace clockwork;
using namespace clockwork::controller;

RequestTelemetryLogger* logger = nullptr;

void signalHandler(int signum) {
	std::cout << "Interrupt signal (" << signum << ") received." << std::endl;
	if (logger != nullptr) { logger->shutdown(false); }
	std::cout << "Clockwork Controller Exiting" << std::endl;
	exit(signum);
}

int main(int argc, char *argv[]) {
	// register signal SIGINT and signal handler
	signal(SIGTERM, signalHandler);
	signal(SIGINT, signalHandler);

	threading::initProcess();	

	std::cout << "Starting Clockwork Controller" << std::endl;

	if ( argc < 5) {
		std::cerr << "USAGE ./controller"
			<< " [CLOSED_LOOP/DIRECT/ECHO/SMPLE/STRESS/INFER]"
			<< " MAX_BATCH_SIZE"
			<< " REQUEST_TELEMETRY_FILE/STDOUT"
			<< " worker1:port1 worker2:port2 ..."
			<< std::endl;
		return 1;
	}
	
	std::string controller_type = argv[1];

	int batch_size = atoi(argv[2]);

	std::string request_telemetry_file = argv[3];

	int client_requests_listen_port = 12346;


	std::vector<std::pair<std::string, std::string>> worker_host_port_pairs;
	for (int i = 4; i < argc; i++) {
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
			ControllerRequestTelemetry::log_and_summarize(
				"/local/clockwork_request_log.tsv",		// 
				10000000000UL 		 	// print request summary every 10s
			)
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
			ControllerRequestTelemetry::log_and_summarize(
				"/local/clockwork_request_log.tsv",		// 
				10000000000UL 		 	// print request summary every 10s
			)
		);
		controller->join();
	} else if (controller_type == "SIMPLE") {
		Scheduler* scheduler = new EchoScheduler(); // TODO
		if (request_telemetry_file == "STDOUT") {
			logger = ControllerRequestTelemetry::summarize(10000000000UL);
		} else {
			logger = ControllerRequestTelemetry::log_and_summarize(
				request_telemetry_file, 10000000000UL);
		}
		controller::ControllerWithStartupPhase* controller =
			new controller::ControllerWithStartupPhase(
			client_requests_listen_port,
			worker_host_port_pairs,
			10000000000UL, // 10s load stage timeout
			new controller::ControllerStartup(),
			scheduler,
			logger
		);
		controller->join();
	}

	std::cout << "Clockwork Controller Exiting" << std::endl;
}
