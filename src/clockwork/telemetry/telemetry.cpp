#include "clockwork/telemetry/controller_request_logger.h"

namespace clockwork {

RequestTelemetryFileLogger::RequestTelemetryFileLogger(std::string filename) : f(filename) {
	write_headers();
}

void RequestTelemetryFileLogger::write_headers() {
	f << "t" << "\t";
	f << "request_id" << "\t";
	f << "result" << "\t";
	f << "user_id" << "\t";
	f << "model_id" << "\t";
	f << "latency" << "\n";
}

void RequestTelemetryFileLogger::log(ControllerRequestTelemetry &t) {
	f << t.departure << "\t";
	f << t.request_id << "\t";
	f << t.result << "\t";
	f << t.user_id << "\t";
	f << t.model_id << "\t";
	f << (t.departure - t.arrival) << "\n";
}

void RequestTelemetryFileLogger::shutdown(bool awaitCompletion) {
	f.close();
}

AsyncRequestTelemetryLogger::AsyncRequestTelemetryLogger() {}

void AsyncRequestTelemetryLogger::addLogger(RequestTelemetryLogger* logger) {
	loggers.push_back(logger);
}

void AsyncRequestTelemetryLogger::start() {
	thread = std::thread(&AsyncRequestTelemetryLogger::run, this);
	threading::initLoggerThread(thread);
}

void AsyncRequestTelemetryLogger::run() {
	while (alive) {
		ControllerRequestTelemetry next;
		while (queue.try_pop(next)) {
			for (auto &logger : loggers) {
				logger->log(next);
			}
		}

		usleep(1000);
	}
}

void AsyncRequestTelemetryLogger::log(ControllerRequestTelemetry &telemetry) {
	queue.push(telemetry);
}

void AsyncRequestTelemetryLogger::shutdown(bool awaitCompletion) {
	alive = false;
	for (auto & logger : loggers) {
		logger->shutdown(true);
	}
}

RequestTelemetryPrinter::RequestTelemetryPrinter(uint64_t print_interval) :
	print_interval(print_interval) {}

void RequestTelemetryPrinter::print(uint64_t interval) {
	if (buffered.size() == 0) {
		std::stringstream ss;
		ss << "Client throughput=0" << std::endl;
		std::cout << ss.str();
		return;
	}

	uint64_t duration_sum = 0;
	unsigned count = 0;
	unsigned violations = 0;
	uint64_t min_latency = UINT64_MAX;
	uint64_t max_latency = 0;
	while (buffered.size() > 0) {
		ControllerRequestTelemetry &next = buffered.front();

		if (next.result == clockworkSuccess) {
			uint64_t latency = (next.departure - next.arrival);
			duration_sum += latency;
			count++;
			min_latency = std::min(min_latency, latency);
			max_latency = std::max(max_latency, latency);
		} else {
			violations++;
		}

		buffered.pop();
	}

	double throughput = (1000000000.0 * count) / ((double) interval);
	double success_rate = 100;
	if (count > 0 || violations > 0) {
		success_rate = count / ((double) (count + violations));
	}

	std::stringstream ss;
	ss << std::fixed;
	if (count == 0) {
		ss << "Client throughput=0 success=0% (" << violations << "/" << violations << " violations)";
	} else {
		ss << "Client throughput=" << std::setprecision(1) << throughput;
		ss << " success=" << std::setprecision(2) << (100*success_rate) << "%";
		if (violations > 0) {
			ss << " (" << violations << "/" << (count+violations) << " violations)";
		}
		ss << " min=" << std::setprecision(1) << (min_latency / 1000000.0);
		ss << " max=" << std::setprecision(1) << (max_latency / 1000000.0);
		ss << " mean=" << std::setprecision(1) << ((duration_sum/count) / 1000000.0);
	}
	ss << std::endl;
	std::cout << ss.str();
}

void RequestTelemetryPrinter::log(ControllerRequestTelemetry &telemetry) {
	buffered.push(telemetry);

	uint64_t now = util::now();
	if (last_print + print_interval <= now) {
		print(now - last_print);
		last_print = now;
	}
}

void RequestTelemetryPrinter::shutdown(bool awaitCompletion) {
	std::cout << "RequestTelemetryPrinter shutting down" << std::endl;
	std::cout << std::flush;
}

RequestTelemetryLogger* ControllerRequestTelemetry::summarize(uint64_t print_interval) {
	auto result = new AsyncRequestTelemetryLogger();
	result->addLogger(new RequestTelemetryPrinter(print_interval));
	result->start();
	return result;
}

RequestTelemetryLogger* ControllerRequestTelemetry::log_and_summarize(std::string filename, uint64_t print_interval) {
	auto result = new AsyncRequestTelemetryLogger();
	result->addLogger(new RequestTelemetryFileLogger(filename));
	result->addLogger(new RequestTelemetryPrinter(print_interval));
	result->start();
	return result;
}

}
