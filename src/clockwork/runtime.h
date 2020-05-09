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

	virtual void executorMain(unsigned executor_id, unsigned core) = 0;
};

class CPUExecutor : public BaseExecutor {
public:
	CPUExecutor(TaskType type, std::vector<unsigned> cores);

	void executorMain(unsigned executor_id, unsigned core);
};

class GPUExecutorShared : public BaseExecutor {
private:
	unsigned num_gpus;

public:
	GPUExecutorShared(TaskType type, std::vector<unsigned> cores, unsigned num_gpus);

	void executorMain(unsigned executor_id, unsigned core);
};

class GPUExecutorExclusive : public BaseExecutor {
private:
	unsigned gpu_id;
	unsigned priority;

public:
	GPUExecutorExclusive(TaskType type, std::vector<unsigned> cores, unsigned gpu_id, int priority = 0);

	void executorMain(unsigned executor_id, unsigned core);
};

class AsyncTaskChecker {
private:
	std::atomic_bool alive;
	tbb::concurrent_queue<AsyncTask*> queue;
	std::vector<std::thread> threads;

public:

	AsyncTaskChecker(std::vector<unsigned> cores);

	void enqueue(AsyncTask* task);
	void shutdown();
	void join();
	void executorMain(unsigned executor_id, unsigned core);
};


class ClockworkRuntime {
public:
	unsigned num_gpus;
	MemoryManager* manager;

	std::vector<GPUExecutorExclusive*> gpu_executors;	// Type 3

	CPUExecutor* load_model_executor;	// Type 0


	std::vector<GPUExecutorExclusive*> weights_executors;	// Type 1
	std::vector<GPUExecutorExclusive*> inputs_executors;		// Type 2
	std::vector<GPUExecutorExclusive*> outputs_executors;	// Type 4


	std::vector<AsyncTaskChecker*> all_checkers; // all checker instances

	std::vector<AsyncTaskChecker*> gpu_checkers;
	std::vector<AsyncTaskChecker*> weights_checkers;
	std::vector<AsyncTaskChecker*> input_checkers;
	std::vector<AsyncTaskChecker*> output_checkers;

	std::vector<CudaEventPool *> event_pools;

	TaskTelemetryLogger* task_telemetry_logger; 
	ActionTelemetryLogger* action_telemetry_logger; 

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

		task_telemetry_logger->shutdown(true);
		action_telemetry_logger->shutdown(true);

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			delete gpu_executors[gpu_id];
			delete weights_executors[gpu_id];
			delete inputs_executors[gpu_id];
			delete outputs_executors[gpu_id];
		}
		for (AsyncTaskChecker* checker : all_checkers) {
			delete checker;
		}
	}

	void shutdown(bool await_completion);

	void join();

protected:

	// Utility class for allocating cores
	class CoreAllocator {
	public:
		std::vector<unsigned> usage_count;
		CoreAllocator() {
			usage_count.resize(util::get_num_cores(), 0);
		}

		int try_acquire(unsigned gpu_id) {
			std::vector<unsigned> preferred = util::get_gpu_core_affinity(gpu_id);
			for (unsigned i = preferred.size()-1; i >= 0; i--) {
				unsigned core = preferred[i];
				if (usage_count[core] == 0) {
					usage_count[core]++;
					return core;
				}
			}
			for (unsigned core = 0; core < usage_count.size(); core++) {
				if (usage_count[core] == 0) {
					usage_count[core]++;
					return core;
				}
			}
			return -1;
		}

		unsigned acquire(unsigned gpu_id) {
			int core = try_acquire(gpu_id);
			CHECK(core >= 0) << "Unable to acquire core for GPU " << gpu_id << "; all cores exhausted";
			return static_cast<unsigned>(core);
		}

	};

	void initialize(ClockworkWorkerConfig &config) {

		num_gpus = config.num_gpus;

		manager = new MemoryManager(config);

		CoreAllocator cores;

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			event_pools.push_back(new CudaEventPool(gpu_id));
			
			gpu_executors.push_back(new GPUExecutorExclusive(GPU, {cores.acquire(gpu_id)}, gpu_id)); // Type 3
			weights_executors.push_back(new GPUExecutorExclusive(PCIe_H2D_Weights, {cores.acquire(gpu_id)}, gpu_id)); // Type 1
			inputs_executors.push_back(new GPUExecutorExclusive(PCIe_H2D_Inputs, {cores.acquire(gpu_id)}, gpu_id, -1)); // Type 2
			outputs_executors.push_back(new GPUExecutorExclusive(PCIe_D2H_Output, {cores.acquire(gpu_id)}, gpu_id, -1)); // Type 4

			AsyncTaskChecker* c1 = new AsyncTaskChecker({cores.acquire(gpu_id)});

			gpu_checkers.push_back(c1);
			weights_checkers.push_back(c1);
			input_checkers.push_back(c1);
			output_checkers.push_back(c1);

			all_checkers.push_back(c1);
		}

		load_model_executor = new CPUExecutor(CPU, {cores.acquire(0)}); // Type 0

		std::string task_file_path = config.telemetry_log_dir + "/" + config.task_telemetry_log_file;
		std::string action_file_path = config.telemetry_log_dir + "/" + config.action_telemetry_log_file;

		if (config.task_telemetry_logging_enabled)
			task_telemetry_logger = new TaskTelemetryFileLogger(task_file_path);
		else
			task_telemetry_logger = new TaskTelemetryDummyLogger();

		if (config.action_telemetry_logging_enabled)
			action_telemetry_logger = new ActionTelemetryFileLogger(action_file_path);
		else
			action_telemetry_logger = new ActionTelemetryDummyLogger();

	}
};


}

#endif
