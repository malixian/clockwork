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
#include "clockwork/telemetry.h"
#include <dmlc/logging.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>


namespace clockwork {

class TelemetryLogger {
public:
	virtual void log(RequestTelemetry* telemetry) = 0;
	virtual void shutdown(bool awaitCompletion) = 0;	
};

class TelemetryFileLogger : public TelemetryLogger {
private:
	const std::string output_filename;
	std::atomic_bool alive;
	std::thread thread;
	tbb::concurrent_queue<RequestTelemetry*> queue;

public:	
	TelemetryFileLogger(std::string output_filename) : output_filename(output_filename), alive(true) {
		thread = std::thread(&TelemetryFileLogger::main, this);
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

	void convert(TaskTelemetry &telemetry, SerializedTaskTelemetry &converted) {
		converted.task_type = telemetry.task_type;
		converted.executor_id = telemetry.executor_id;
		converted.created = util::nanos(telemetry.created);
		converted.enqueued = util::nanos(telemetry.enqueued);
		converted.eligible_for_dequeue = util::nanos(telemetry.eligible_for_dequeue);
		converted.dequeued = util::nanos(telemetry.dequeued);
		converted.exec_complete = util::nanos(telemetry.exec_complete);
		converted.async_complete = util::nanos(telemetry.async_complete);
		converted.async_wait = telemetry.async_wait * 1000000;
		converted.async_duration = telemetry.async_duration * 1000000;
	}

	void convert(RequestTelemetry* telemetry, SerializedRequestTelemetry &converted) {
		converted.model_id = telemetry->model_id;
		converted.request_id = telemetry->request_id;
		converted.arrived = util::nanos(telemetry->arrived);
		converted.submitted = util::nanos(telemetry->submitted);
		converted.complete = util::nanos(telemetry->complete);
		converted.tasks.resize(telemetry->tasks.size());
		for (unsigned i = 0; i < telemetry->tasks.size(); i++) {
			convert(*telemetry->tasks[i], converted.tasks[i]);
		}
	}

	void main() {
		std::ofstream outfile;
		outfile.open(output_filename);
	    pods::OutputStream out(outfile);
	    pods::BinarySerializer<decltype(out)> serializer(out);

		while (alive) {
			RequestTelemetry* srcTelemetry;
			if (!queue.try_pop(srcTelemetry)) {
				usleep(10000);
				continue;
			}

			SerializedRequestTelemetry telemetry;
			convert(srcTelemetry, telemetry);
			delete srcTelemetry;


		    CHECK(serializer.save(telemetry) == pods::Error::NoError) << "Unable to serialize telemetry";

			// std::stringstream msg;
			// msg << "Logging request " << telemetry.request_id 
			//     << " model=" << telemetry.model_id
			//     << " latency=" << (telemetry.complete - telemetry.submitted)
			//     << " totallatency=" << (telemetry.complete - telemetry.arrived)
			//     << std::endl;
			// for (int i = 0; i < telemetry.tasks.size(); i++) {
			// 	msg << "   Task " << i << std::endl
			// 	    << "     queue=" << (telemetry.tasks[i].dequeued - telemetry.tasks[i].enqueued) << std::endl
			// 	    << "     exec="  << (telemetry.tasks[i].exec_complete - telemetry.tasks[i].dequeued) << std::endl
			// 	    << "     async_wait=" << telemetry.tasks[i].async_wait << std::endl
			// 	    << "     async_duration=" << telemetry.tasks[i].async_duration << std::endl;
			// }
			// std::cout << msg.str();
		}
		outfile.close();
	}

};

}

#endif
