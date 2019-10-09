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

	ClockworkRuntime() {
		// TODO: factor this out; for now hard-coded 8GB cache w/ 16MB pages
		int page_size = 16 * 1024 * 1024;
		int cache_size = 50 * page_size;
		void* baseptr;
    	CUDA_CALL(cudaMalloc(&baseptr, cache_size));
    	PageCache* cache = new PageCache(static_cast<char*>(baseptr), cache_size, page_size, false);

		manager = new MemoryManager();
	    manager->weights_cache = cache;
	    manager->workspace_cache = cache;
	    manager->models = new ModelStore();

	    load_model_executor = new Executor(CPU);
		weights_executor = new Executor(PCIe_H2D_Weights);
		inputs_executor = new Executor(PCIe_H2D_Inputs);
		gpu_executor = new Executor(GPU);
		outputs_executor = new Executor(PCIe_D2H_Output);
		checker = new AsyncTaskChecker();
	}

	~ClockworkRuntime() {
		delete manager->models;
		CUDA_CALL(cudaFree(manager->weights_cache->baseptr));
		delete manager->weights_cache;
		delete load_model_executor;
		delete weights_executor;
		delete inputs_executor;
		delete gpu_executor;
		delete outputs_executor;
		delete checker;
	}

	void shutdown(bool await_completion) {
		load_model_executor->shutdown();
		weights_executor->shutdown();
		inputs_executor->shutdown();
		gpu_executor->shutdown();
		outputs_executor->shutdown();
		checker->shutdown();
		if (await_completion) {
			join();
		}
	}

	void join() {
		load_model_executor->join();
		weights_executor->join();
		inputs_executor->join();
		gpu_executor->join();
		outputs_executor->join();
		checker->join();
	}

};


}

#endif
