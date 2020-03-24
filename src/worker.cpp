#include "clockwork/worker.h"
#include "clockwork/network/worker.h"
#include "clockwork/runtime.h"


int main(int argc, char *argv[]) {
	std::cout << "Starting Clockwork Worker" << std::endl;

	clockwork::ClockworkWorkerSettings settings = clockwork::ClockworkWorkerSettings();

	for (int i = 1; i < argc; i++){
		if ((strcmp(argv[i], "--task") == 0) || (strcmp(argv[i], "-t") == 0))
			settings.task_telemetry_logging_enabled = true;
		if ((strcmp(argv[i], "--action") == 0) || (strcmp(argv[i], "-a") == 0))
			settings.action_telemetry_logging_enabled = true;
		if (strcmp(argv[i], "--task_log_dir") == 0) {
			settings.task_telemetry_logging_enabled = true;
			settings.task_telemetry_log_dir = argv[i+1];
		}
		if (strcmp(argv[i], "--action_log_dir") == 0) {
			settings.action_telemetry_logging_enabled = true;
			settings.action_telemetry_log_dir = argv[i+1];
		}
	}

	clockwork::ClockworkWorker* clockwork = new clockwork::ClockworkWorker(settings);
	clockwork::network::worker::Server* server = new clockwork::network::worker::Server(clockwork);
	clockwork->controller = server;
	clockwork->join();

	std::cout << "Clockwork Worker Exiting" << std::endl;
}
