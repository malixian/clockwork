#ifndef _CLOCKWORK_RUNTIME_H_
#define _CLOCKWORK_RUNTIME_H_

#include <thread>
#include <limits>
#include <algorithm>
#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/cuda_common.h"
#include "clockwork/telemetry.h"
#include "../src/clockwork/telemetry/task_telemetry_logger.h"
#include "../src/clockwork/telemetry/action_telemetry_logger.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/priority_queue.h"
#include "clockwork/common.h"
#include "tbb/concurrent_queue.h"
#include "clockwork/task.h"
#include "clockwork/memory.h"
#include "clockwork/config.h"

/*
This file contains the clockwork scheduling and thread pool logic for executing tasks, asynchronous
tasks, and checking async task completion.
*/

namespace clockwork {

class ClockworkRuntime;

class BaseExecutor {
public:
	const TaskType type;
	std::atomic_bool alive;
	std::vector<std::thread> threads;
	single_reader_priority_queue<Task> queue;

	BaseExecutor(TaskType type) : type(type), alive(true) {}

	void enqueue(Task* task);
	void shutdown();
	void join();

	virtual void executorMain(unsigned executor_id) = 0;
};

class CPUExecutor : public BaseExecutor {
public:
	CPUExecutor(TaskType type);

	void executorMain(unsigned executor_id);
};

class SingleThreadExecutor : public BaseExecutor {
private:
	unsigned gpu_id;
	tbb::concurrent_queue<AsyncTask*> runqueue;

public:
	SingleThreadExecutor(TaskType type, unsigned gpu_id);
	void enqueue(AsyncTask* task);
	void executorMain(unsigned executor_id);
};


class ClockworkRuntime {
public:
	unsigned num_gpus;
	MemoryManager* manager;// TODO WEI
	util::GPUClockState* gpu_clock;

	std::vector<SingleThreadExecutor*> executors;	// Type 3

	CPUExecutor* load_model_executor;	// Type 0

	std::vector<CudaEventPool *> event_pools;// TODO WEI

	//TaskTelemetryLogger* task_telemetry_logger; 
	//ActionTelemetryLogger* action_telemetry_logger; 

	ClockworkRuntime() {
		ClockworkWorkerConfig config;
		initialize(config);
	}

	ClockworkRuntime(ClockworkWorkerConfig &config) {
		initialize(config);
	}

	virtual ~ClockworkRuntime() {
		delete manager;
		delete load_model_executor;

		//task_telemetry_logger->shutdown(true);
		//action_telemetry_logger->shutdown(true);

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			delete executors[gpu_id];

		}
	}

	void shutdown(bool await_completion);

	void join();

protected:


	void initialize(ClockworkWorkerConfig &config) {

		num_gpus = config.num_gpus;

		gpu_clock = new util::GPUClockState(num_gpus);

		manager = new MemoryManager(config);// TODO WEI

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			event_pools.push_back(new CudaEventPool(gpu_id));
			
			executors.push_back(new SingleThreadExecutor(GPU, gpu_id)); // Type 3

		}

		load_model_executor = new CPUExecutor(CPU); // Type 0

		std::string task_file_path = config.telemetry_log_dir + "/" + config.task_telemetry_log_file;
		std::string action_file_path = config.telemetry_log_dir + "/" + config.action_telemetry_log_file;
		/*
		if (config.task_telemetry_logging_enabled)
			task_telemetry_logger = new TaskTelemetryFileLogger(task_file_path);
		else
			task_telemetry_logger = new TaskTelemetryDummyLogger();

		if (config.action_telemetry_logging_enabled)
			action_telemetry_logger = new ActionTelemetryFileLogger(action_file_path);
		else
			action_telemetry_logger = new ActionTelemetryDummyLogger();*/

	}
};


}

#endif
