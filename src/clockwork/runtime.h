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

class Executor {
private:
	std::atomic_bool alive;
	time_release_priority_queue<Task> queue;
	std::vector<std::thread> threads;

public:
	const TaskType type;

	Executor(TaskType type, int num_threads = 1);

	void enqueue(Task* task);
	void shutdown();
	void join();
	void executorMain(int executorId);
};

class AsyncTaskChecker {
private:
	std::atomic_bool alive;
	tbb::concurrent_queue<AsyncTask*> queue;
	std::vector<std::thread> threads;

public:

	AsyncTaskChecker();

	void enqueue(AsyncTask* task);
	void shutdown();
	void join();
	void executorMain(int executorId);
};

class ClockworkRuntime {
public:
	MemoryManager* manager;

	Executor* load_model_executor;
	Executor* weights_executor;
	Executor* inputs_executor;
	Executor* gpu_executor;
	Executor* outputs_executor;

	AsyncTaskChecker* checker;

	// TODO: currently we've hard-coded a whole bunch of defaults -- 10GB cache, 16MB pages

	ClockworkRuntime() {
		size_t weights_page_size = 16L * 1024L * 1024L; // 16MB hard-coded weights cache page size
		size_t weights_cache_size = 10L * 1024L * 1024L * 1024L; // 10 GB hard-coded weights cache for now
		size_t workspace_page_size = 64L * 1024L * 1024L; // 64MB hard-coded workspace cache page size
		size_t workspace_cache_size = 512L * 1024L * 1024L; // Shouldn't need too much workspace cache
		PageCache* weights_cache = make_GPU_cache(weights_cache_size, weights_page_size);
		PageCache* workspace_cache = make_GPU_cache(workspace_cache_size, workspace_page_size);
		manager = new MemoryManager(weights_cache, workspace_cache);

	    load_model_executor = new Executor(CPU);
		weights_executor = new Executor(PCIe_H2D_Weights);
		inputs_executor = new Executor(PCIe_H2D_Inputs);
		gpu_executor = new Executor(GPU);
		outputs_executor = new Executor(PCIe_D2H_Output);
		checker = new AsyncTaskChecker();
	}

	virtual ~ClockworkRuntime() {
		delete manager;
		delete load_model_executor;
		delete weights_executor;
		delete inputs_executor;
		delete gpu_executor;
		delete outputs_executor;
		delete checker;
	}

	void shutdown(bool await_completion);

	void join();

};


}

#endif
