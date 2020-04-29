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
	time_release_priority_queue<Task> queue;

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

public:
	GPUExecutorExclusive(TaskType type, std::vector<unsigned> cores, unsigned gpu_id);

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

	GPUExecutorShared* weights_executor;	// Type 1
	GPUExecutorShared* inputs_executor;		// Type 2
	GPUExecutorShared* outputs_executor;	// Type 4

	AsyncTaskChecker* checker;

	std::vector<CudaEventPool *> event_pools;

	TaskTelemetryLogger* task_telemetry_logger; 
	ActionTelemetryLogger* action_telemetry_logger; 

	ClockworkRuntime() {
		initialize(ClockworkWorkerSettings());
	}

	ClockworkRuntime(ClockworkWorkerSettings settings) {
		initialize(settings);
	}

	virtual ~ClockworkRuntime() {
		delete manager;
		delete load_model_executor;
		delete weights_executor;
		delete inputs_executor;
		delete outputs_executor;
		delete checker;

		task_telemetry_logger->shutdown(true);
		action_telemetry_logger->shutdown(true);

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			delete gpu_executors[gpu_id];
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

	void initialize(ClockworkWorkerSettings settings) {

		num_gpus = settings.num_gpus;

		manager = new MemoryManager(settings);

		CoreAllocator cores;

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			event_pools.push_back(new CudaEventPool(gpu_id));
			
			gpu_executors.push_back(new GPUExecutorExclusive(GPU, {cores.acquire(gpu_id)}, gpu_id)); // Type 3

			if (gpu_id == 0) {
				load_model_executor = new CPUExecutor(CPU, {cores.acquire(gpu_id)}); // Type 0
				weights_executor = new GPUExecutorShared(PCIe_H2D_Weights, {cores.acquire(gpu_id)}, num_gpus);	// Type 1
				inputs_executor = new GPUExecutorShared(PCIe_H2D_Inputs, {cores.acquire(gpu_id)}, num_gpus);	// Type 2
				outputs_executor = new GPUExecutorShared(PCIe_D2H_Output, {cores.acquire(gpu_id)}, num_gpus);	// Type 4
				checker = new AsyncTaskChecker({cores.acquire(gpu_id)});
			}
		}
		std::string task_file_name = settings.task_telemetry_log_dir;
		std::string action_file_name = settings.action_telemetry_log_dir;

		if (settings.task_telemetry_logging_enabled)
			task_telemetry_logger = new TaskTelemetryFileLogger(task_file_name);
		else
			task_telemetry_logger = new TaskTelemetryDummyLogger();

		if (settings.action_telemetry_logging_enabled)
			action_telemetry_logger = new ActionTelemetryFileLogger(action_file_name);
		else
			action_telemetry_logger = new ActionTelemetryDummyLogger();

	}
};


}

#endif
