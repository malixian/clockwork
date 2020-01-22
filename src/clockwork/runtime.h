#ifndef _CLOCKWORK_RUNTIME_H_
#define _CLOCKWORK_RUNTIME_H_

#include <thread>
#include <limits>
#include <algorithm>
#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/telemetry.h"
#include "clockwork/telemetry_logger.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/priority_queue.h"
#include "clockwork/common.h"
#include "tbb/concurrent_queue.h"
#include "clockwork/task.h"

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

	TelemetryFileLogger* telemetry_logger; 

	// TODO: currently we've hard-coded a whole bunch of defaults -- 10GB cache, 16MB pages

	ClockworkRuntime() {
		size_t weights_cache_size = 10L * 1024L * 1024L * 1024L; // 10 GB hard-coded weights cache for now
		size_t weights_cache_page_size = 16L * 1024L * 1024L;	 // 16MB hard-coded weights cache page size
		size_t io_pool_size = 128L * 1024L * 1024L;				 // 128 MB hard-coded io pool size
		size_t workspace_pool_size = 512L * 1024L * 1024L;		 // 512 MB hard-coded workspace pool size
		size_t host_io_pool_size = 256L * 1024L * 1024L;		 // 256 MB hard-coded host IO pool size

		num_gpus = util::get_num_gpus();

		manager = new MemoryManager(
			weights_cache_size, weights_cache_page_size,
			io_pool_size,
			workspace_pool_size,
			host_io_pool_size,
			num_gpus
		);

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			event_pools.push_back(new CudaEventPool(gpu_id));

			auto cores = util::get_gpu_core_affinity(gpu_id);
			int i = cores.size()-1;

			gpu_executors.push_back(new GPUExecutorExclusive(GPU, {cores[i--]}, gpu_id)); // Type 3

			if (gpu_id == 0) {
				load_model_executor = new CPUExecutor(CPU, {cores[i--]}); // Type 0
				weights_executor = new GPUExecutorShared(PCIe_H2D_Weights, {cores[i--]}, num_gpus);	// Type 1
				inputs_executor = new GPUExecutorShared(PCIe_H2D_Inputs, {cores[i--]}, num_gpus);	// Type 2
				outputs_executor = new GPUExecutorShared(PCIe_D2H_Output, {cores[i--]}, num_gpus);	// Type 4
				checker = new AsyncTaskChecker({cores[i--]});
			}
		}

		telemetry_logger = new TelemetryFileLogger("telemetry.raw");
	}

	virtual ~ClockworkRuntime() {
		delete manager;
		delete load_model_executor;
		delete weights_executor;
		delete inputs_executor;
		delete outputs_executor;
		delete checker;

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			delete gpu_executors[gpu_id];
		}
	}

	void shutdown(bool await_completion);

	void join();

};


}

#endif
