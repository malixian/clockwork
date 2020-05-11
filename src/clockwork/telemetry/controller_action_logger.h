#ifndef _CLOCKWORK_TELEMETRY_CONTROLLER_ACTION_LOGGER_H_
#define _CLOCKWORK_TELEMETRY_CONTROLLER_ACTION_LOGGER_H_

#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"
#include <algorithm>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <numeric>
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
#include <tuple>
#include "clockwork/api/worker_api.h"


namespace clockwork {

struct ControllerActionTelemetry {
	int action_id;
	int worker_id;
	int gpu_id;
	int action_type;
	int batch_size;
	int model_id;
	uint64_t earliest;
	uint64_t latest;
	uint64_t action_sent;
	uint64_t result_received;
	int status;
	uint64_t worker_duration;
};

class ControllerActionTelemetryLogger {
public:
	virtual void log(ControllerActionTelemetry &telemetry) = 0;
	virtual void shutdown(bool awaitCompletion) = 0;
};

class NoOpControllerActionTelemetryLogger : public ControllerActionTelemetryLogger {
public:
	virtual void log(ControllerActionTelemetry &telemetry) {}
	virtual void shutdown(bool awaitCompletion) {}
};

class AsyncControllerActionTelemetryLogger : public ControllerActionTelemetryLogger {
private:
	std::atomic_bool alive = true;
	std::thread thread;
	tbb::concurrent_queue<ControllerActionTelemetry> queue;
	std::vector<ControllerActionTelemetryLogger*> loggers;

public:	

	AsyncControllerActionTelemetryLogger() {}

	void addLogger(ControllerActionTelemetryLogger* logger) {
		loggers.push_back(logger);
	}

	void start() {
		this->thread = std::thread(&AsyncControllerActionTelemetryLogger::run, this);
	}

	void run() {
		while (alive) {
			ControllerActionTelemetry next;
			while (queue.try_pop(next)) {
				for (auto &logger : loggers) {
					logger->log(next);
				}
			}

			usleep(1000);
		}
	}

	void log(ControllerActionTelemetry &telemetry) {
		queue.push(telemetry);
	}

	void shutdown(bool awaitCompletion) {
		alive = false;
		for (auto & logger : loggers) {
			logger->shutdown(true);
		}
	}
};

class ActionPrinter : public ControllerActionTelemetryLogger {
private:
	uint64_t last_print;
	const uint64_t print_interval;
	std::queue<ControllerActionTelemetry> buffered;

public:

	ActionPrinter(uint64_t print_interval) : print_interval(print_interval) {}

	virtual void print(uint64_t interval, std::queue<ControllerActionTelemetry> &buffered) = 0;

	void log(ControllerActionTelemetry &telemetry) {
		buffered.push(telemetry);

		uint64_t now = util::now();
		if (last_print + print_interval <= now) {
			print(now - last_print, buffered);
			last_print = now;
		}
	}

	void shutdown(bool awaitCompletion) {}

};

class SimpleActionPrinter : public ActionPrinter {
public:

	SimpleActionPrinter(uint64_t print_interval) : ActionPrinter(print_interval) {}

	static AsyncControllerActionTelemetryLogger* create_async(uint64_t print_interval) {
		auto result = new AsyncControllerActionTelemetryLogger();
		result->addLogger(new SimpleActionPrinter(print_interval));
		result->start();
		return result;
	}

	typedef std::tuple<int,int,int> Group;

	std::map<Group, std::queue<ControllerActionTelemetry>> make_groups(
			std::queue<ControllerActionTelemetry> &buffered) {
		std::map<Group, std::queue<ControllerActionTelemetry>> result;
		while (!buffered.empty()) {
			auto &t = buffered.front();
			if ((t.action_type == workerapi::loadWeightsAction || 
				t.action_type == workerapi::inferAction) &&
				t.status == clockworkSuccess) {
				Group key = std::make_tuple(t.worker_id, t.gpu_id, t.action_type);
				result[key].push(t);
				buffered.pop();
			}
		}
		return result;
	}

	struct Stat {
		std::vector<uint64_t> v;

		unsigned size() { return v.size(); }
		uint64_t min() { return *std::min_element(v.begin(), v.end()); }
		uint64_t max() { return *std::max_element(v.begin(), v.end()); }
		uint64_t mean() { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }
		double throughput(uint64_t interval) {
			return (size() * 1000000000.0) / static_cast<double>(interval);
		}
		double utilization(uint64_t interval) { 
			return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(interval);
		}
	};

	void print(uint64_t interval, const Group &group, std::queue<ControllerActionTelemetry> &buffered) {
		if (buffered.empty()) return;

		int worker_id = std::get<0>(group);
		int gpu_id = std::get<1>(group);
		int action_type = std::get<2>(group);

		Stat e2e;
		Stat w;

		if (buffered.empty()) {
			std::stringstream s;
			s << std::fixed << std::setprecision(2);
			s << "W" << worker_id
			  << "-GPU" << gpu_id
			  << " throughput=0" << std::endl;
			std::cout << s.str();
			return;
		}

		while (!buffered.empty()) {
			auto &t = buffered.front();
			e2e.v.push_back(t.result_received - t.action_sent);
			w.v.push_back(t.worker_duration);
			buffered.pop();
		}

		std::stringstream s;
		s << std::fixed << std::setprecision(2);
		s << "W" << worker_id
		  << "-GPU" << gpu_id;

		switch(action_type) {
			case workerapi::loadWeightsAction: s << " LoadW"; break;
			case workerapi::inferAction: s << " Infer"; break;
			default: return;
		}

		s << " min=" << w.min() 
		  << " max=" << w.max() 
		  << " mean=" << w.mean() 
		  << " e2emean=" << e2e.mean() 
		  << " e2emax=" << e2e.max() 
		  << " throughput=" << w.throughput(interval) 
		  << " utilization=" << w.utilization(interval)
		  << std::endl;
		std::cout << s.str();
	}

	void print(uint64_t interval, std::queue<ControllerActionTelemetry> &buffered) {
		auto groups = make_groups(buffered);

		for (auto &p : groups) {
			print(interval, p.first, p.second);
		}
	}

};

}

#endif