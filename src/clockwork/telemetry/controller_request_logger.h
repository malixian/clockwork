#ifndef _CLOCKWORK_TELEMETRY_CONTROLLER_REQUEST_LOGGER_H_
#define _CLOCKWORK_TELEMETRY_CONTROLLER_REQUEST_LOGGER_H_

#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"
#include <algorithm>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include "clockwork/util.h"
#include "clockwork/telemetry.h"
#include <dmlc/logging.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>
#include <tbb/concurrent_queue.h>
#include <iomanip>
#include "clockwork/api/api_common.h"


namespace clockwork {

struct ControllerRequestTelemetry {
	int request_id;
	int user_id;
	int model_id;
	uint64_t arrival;
	uint64_t departure;
	int result;
};

class RequestTelemetryLogger {
public:
	virtual void log(ControllerRequestTelemetry &telemetry) = 0;
	virtual void shutdown(bool awaitCompletion) = 0;
};

class NoOpRequestTelemetryLogger : public RequestTelemetryLogger {
public:
	virtual void log(ControllerRequestTelemetry &telemetry) {}
	virtual void shutdown(bool awaitCompletion) {}
};

class AsyncRequestTelemetryLogger : public RequestTelemetryLogger {
private:
	std::atomic_bool alive = true;
	std::thread thread;
	tbb::concurrent_queue<ControllerRequestTelemetry> queue;
	std::vector<RequestTelemetryLogger*> loggers;

public:	

	AsyncRequestTelemetryLogger() {}

	void addLogger(RequestTelemetryLogger* logger) {
		loggers.push_back(logger);
	}

	void start() {
		thread = std::thread(&AsyncRequestTelemetryLogger::run, this);
	}

	void run() {
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

	void log(ControllerRequestTelemetry &telemetry) {
		queue.push(telemetry);
	}

	void shutdown(bool awaitCompletion) {
		alive = false;
		for (auto & logger : loggers) {
			logger->shutdown(true);
		}
	}
};

class RequestTelemetryPrinter : public RequestTelemetryLogger {
private:
	uint64_t last_print;
	const uint64_t print_interval;
	std::queue<ControllerRequestTelemetry> buffered;

public:

	RequestTelemetryPrinter(uint64_t print_interval) : print_interval(print_interval) {}

	static RequestTelemetryLogger* async_request_printer(uint64_t print_interval) {
		auto result = new AsyncRequestTelemetryLogger();
		result->addLogger(new RequestTelemetryPrinter(print_interval));
		result->start();
		return result;
	}

	void print(uint64_t interval) {
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
		ss << "Client throughput=" << std::setprecision(1) << throughput;
		ss << " success=" << std::setprecision(3) << success_rate << "%";
		if (violations > 0) {
			std::cout << "(" << violations << "/" << (count+violations) << " violations)";
		}
		ss << " min=" << std::setprecision(1) << (min_latency / 1000000.0);
		ss << " max=" << std::setprecision(1) << (max_latency / 1000000.0);
		ss << " mean=" << std::setprecision(1) << ((duration_sum/count) / 1000000.0);
		ss << std::endl;
		std::cout << ss.str();
	}

	void log(ControllerRequestTelemetry &telemetry) {
		buffered.push(telemetry);

		uint64_t now = util::now();
		if (last_print + print_interval <= now) {
			print(now - last_print);
			last_print = now;
		}
	}

	void shutdown(bool awaitCompletion) {}

};

}

#endif