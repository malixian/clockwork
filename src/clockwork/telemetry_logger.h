#ifndef _CLOCKWORK_TELEMETRY_LOGGER_H_
#define _CLOCKWORK_TELEMETRY_LOGGER_H_

#include <thread>
#include <atomic>
#include "tbb/concurrent_queue.h"
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include "clockwork/util.h"


namespace clockwork {

class TelemetryLogger {
private:
	const std::string output_filename;
	std::atomic_bool alive = true;
	std::thread thread;
	tbb::concurrent_queue<RequestTelemetry*> queue;

public:	
	TelemetryLogger(std::string output_filename) : output_filename(output_filename) {
		thread = std::thread(&TelemetryLogger::main, this);
	}

	void shutdown(bool awaitCompletion) {
		alive = false;
		if (awaitCompletion) {
			thread.join();
		}
	}

	void log(RequestTelemetry* telemetry) {
		queue.push(telemetry);
	}

	void convert(RequestTelemetry* telemetry, SerializedRequestTelemetry &converted) {
		converted.model_id = telemetry->model_id;
		converted.request_id = telemetry->request_id;
		converted.arrived = util::nanos(telemetry->arrived);
		converted.submitted = util::nanos(telemetry->submitted);
		converted.complete = util::nanos(telemetry->complete);
	}

	void main() {
		std::ofstream out(output_filename);
		while (alive) {
			RequestTelemetry* srcTelemetry;
			if (!queue.try_pop(srcTelemetry)) {
				usleep(10000);
				continue;
			}

			SerializedRequestTelemetry telemetry;
			convert(srcTelemetry, telemetry);
			delete srcTelemetry;

			std::stringstream msg;
			msg << "Logging request " << telemetry.request_id 
			    << " model=" << telemetry.model_id
			    << " latency=" << (telemetry.complete - telemetry.submitted)
			    << " totallatency=" << (telemetry.complete - telemetry.arrived)
			    << std::endl;
			std::cout << msg.str();
		}
		out.close();
	}

};

}

#endif