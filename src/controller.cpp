#include "clockwork/network/controller.h"
#include "clockwork/controller/direct_controller.h"
#include "clockwork/controller/closed_loop_controller.h"
#include "clockwork/controller/stress_test_controller.h"
#include "clockwork/controller/infer_only_scheduler.h"
#include "clockwork/telemetry/controller_request_logger.h"
#include <csignal>
#include <sstream>
#include <string>
#include <vector>
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


void printUsage() {
    std::cerr << "Usage: controller [CONTROLLERTYPE] [workers] [options] "
              << "[workload parameters (if required)]" << std::endl
              << "Available workloads with parameters:" << std::endl
              << "  example" << std::endl
              << "  spam" << std::endl
              << "  simple" << std::endl
              << "  simple-parametric num_models concurrency requests_per_model"
              << std::endl
              << "  azure" << std::endl
              << "  azure_small" << std::endl;
}


void show_usage() {
    std::stringstream s;
    s << "USAGE:\n";
    s << "  controller [TYPE] [WORKERS] [OPTIONS]\n";
    s << "DESCRIPTION\n";
    s << "  Run the controller of the given TYPE. Connects to the specified workers. All  \n";
    s << "  subsequent options are controller-specific and passed to that controller.     \n";
    s << "TYPE\n";
    s << "  CLOSED_LOOP\n";
    s << "  DIRECT\n";
    s << "  ECHO\n";
    s << "  SIMPLE\n";
    s << "  STRESS\n";
    s << "  INFER\n";
    s << "WORKERS\n";
    s << "  Comma-separated list of worker host:port pairs.  e.g.:                        \n";
    s << "    volta03:12345,volta04:12345,volta05:12345                                   \n";
    s << "OPTIONS\n";
    s << "  -h,  --help\n";
    s << "        Print this message\n";
    s << "All other options are passed to the specific scheduler on init\n";
    std::cout << s.str();
}

std::vector<std::string> split(std::string string, char delimiter = ',') {
    std::stringstream ss(string);
    std::vector<std::string> result;

    while( ss.good() )
    {
        std::string substr;
        getline( ss, substr, delimiter);
        result.push_back( substr );
    }
    return result;
}

int main(int argc, char *argv[]) {
    if ( argc < 3) {
        show_usage();
        return 1;
    }

    // register signal SIGINT and signal handler
    signal(SIGTERM, signalHandler);
    signal(SIGINT, signalHandler);

    threading::initProcess();

    std::cout << "Starting Clockwork Controller" << std::endl;
    
    std::string controller_type = argv[1];

    std::vector<std::string> workers = split(argv[2]);
    std::vector<std::pair<std::string, std::string>> worker_host_port_pairs;
    for (std::string worker : workers) {
        std::vector<std::string> p = split(worker,':');
        worker_host_port_pairs.push_back({p[0], p[1]});
    }

    int client_requests_listen_port = 12346;

    if ( controller_type == "CLOSED_LOOP"){
        int batch_size = atoi(argv[3]);
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
            100000000UL, // 10s load stage timeout
            new controller::ControllerStartup(), // in future the startup type might be customizable
            scheduler,
            ControllerRequestTelemetry::log_and_summarize(
                "/local/clockwork_request_log.tsv",     // 
                10000000000UL           // print request summary every 10s
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
                "/local/clockwork_request_log.tsv",     // 
                10000000000UL           // print request summary every 10s
            )
        );
        controller->join();
    } else if (controller_type == "SIMPLE") {
        Scheduler* scheduler = new EchoScheduler(); // TODO
        std::string request_telemetry_file = argv[3];
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
    } else {
        std::cout << "Invalid controller type " << controller_type << std::endl;
        show_usage();
    }

    std::cout << "Clockwork Controller Exiting" << std::endl;
}
