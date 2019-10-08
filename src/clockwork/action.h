#ifndef _CLOCKWORK_ACTION_H_
#define _CLOCKWORK_ACTION_H_

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

namespace clockwork {

class Executor {
private:
	std::atomic_bool alive;
	time_release_priority_queue<Task> queue;
	std::vector<std::thread> threads;


public:
	const TaskType type;

	Executor(TaskType type);

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

		weights_executor = new Executor(PCIe_H2D_Weights);
		inputs_executor = new Executor(PCIe_H2D_Inputs);
		gpu_executor = new Executor(GPU);
		outputs_executor = new Executor(PCIe_D2H_Output);
		checker = new AsyncTaskChecker();
	}

};

class Action {
public:
	virtual void submit() = 0;
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;
};

class LoadWeightsAction : public Action {
private:

	class LoadWeightsTaskImpl : public LoadWeightsTask {
	public:
		LoadWeightsAction* action;

		LoadWeightsTaskImpl(LoadWeightsAction* action);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm);
		void error(int status_code, std::string message);
	};

	ClockworkRuntime* runtime;
	int model_id;
	uint64_t earliest, latest;
	LoadWeightsTaskImpl* task;

public:
	LoadWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest);
	~LoadWeightsAction();
	void submit();
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;
};

class EvictWeightsAction : public Action {
private:

	class EvictWeightsTaskImpl : public EvictWeightsTask {
	public:
		EvictWeightsAction* action;

		EvictWeightsTaskImpl(EvictWeightsAction* action);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm);
		void error(int status_code, std::string message);
	};

	ClockworkRuntime* runtime;
	int model_id;
	uint64_t earliest, latest;
	EvictWeightsTaskImpl* task;

public:
	EvictWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest);
	~EvictWeightsAction();
	void submit();
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;
};


class InferAction : public Action {
private:

	class CopyOutputTaskImpl : public CopyOutputTask {
	public:
		InferAction* action;

		CopyOutputTaskImpl(InferAction* action);

		void run(cudaStream_t stream);
		void success();
		void error(int status_code, std::string message);
	};


	class InferTaskImpl : public InferTask {
	public:
		InferAction* action;

		InferTaskImpl(InferAction* action);

		void run(cudaStream_t stream);
		void success();
		void error(int status_code, std::string message);
	};


	class CopyInputTaskImpl : public CopyInputTask {
	public:
		InferAction* action;

		CopyInputTaskImpl(InferAction* action);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm, std::shared_ptr<Allocation> workspace);
		void error(int status_code, std::string message);
	};

	ClockworkRuntime* runtime;

	int model_id;
	uint64_t earliest, latest;
	char* input;
	char* output;

	RuntimeModel* rm;
	std::shared_ptr<Allocation> workspace;

	CopyInputTaskImpl* copy_input = nullptr;
	InferTaskImpl* infer = nullptr;
	CopyOutputTaskImpl* copy_output = nullptr;
	

	uint64_t copy_input_earliest();

public:
	InferAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest, char* input, char* output);
	~InferAction();

	void submit();
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;

};



}

#endif
