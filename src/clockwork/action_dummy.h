#ifndef _CLOCKWORK_ACTION_H_
#define _CLOCKWORK_ACTION_H_

#include <thread>
#include <limits>
#include <algorithm>
#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/cuda_common.h"
#include "clockwork/telemetry.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/priority_queue.h"
#include "clockwork/common.h"
#include "tbb/concurrent_queue.h"
#include "clockwork/task.h"
#include "clockwork/runtime_dummy.h"
#include "clockwork/api/worker_api.h"

/*
This file defines how to execute tasks (defined in task.h) within the clockwork scheduling
and thread-pool framework (defined in runtime.h).
*/

namespace clockwork {

class LoadModelFromDiskActionDummy {
protected:

	class LoadModelFromDiskTaskImpl : public LoadModelFromDiskTask {
	public:
		LoadModelFromDiskActionDummy* load_model;

		LoadModelFromDiskTaskImpl(LoadModelFromDiskActionDummy* load_model);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm);
		void cancel();
		int duplicate(RuntimeModel* rm, int model_id, int no_of_copies);
	};

	ClockworkRuntimeDummy* runtime;
	std::shared_ptr<workerapi::LoadModelFromDisk> action;
	LoadModelFromDiskTaskImpl* task;

public:
	LoadModelFromDiskActionDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::LoadModelFromDisk> action);
	~LoadModelFromDiskActionDummy();

	void submit();
	void handle_error(TaskError &error);

	virtual void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) = 0;
	virtual void error(std::shared_ptr<workerapi::ErrorResult> result) = 0;
};

class LoadWeightsActionDummy {
protected:

	class LoadWeightsTaskImpl : public LoadWeightsTask {
	public:
		LoadWeightsActionDummy* load_weights;

		LoadWeightsTaskImpl(LoadWeightsActionDummy* load_weights);

		void run(cudaStream_t stream);
		void process_completion();
		void success(RuntimeModel* rm);
		void cancel();
	};

	ClockworkRuntimeDummy* runtime;
	std::shared_ptr<workerapi::LoadWeights> action;
	LoadWeightsTaskImpl* task;

public:
	LoadWeightsActionDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::LoadWeights> action);
	~LoadWeightsActionDummy();

	void submit();
	void handle_error(TaskError &error);

	virtual void success(std::shared_ptr<workerapi::LoadWeightsResult> result) = 0;
	virtual void error(std::shared_ptr<workerapi::ErrorResult> result) = 0;
};

class EvictWeightsActionDummy {
protected:

	class EvictWeightsTaskImpl : public EvictWeightsTask {
	public:
		EvictWeightsActionDummy* evict_weights;

		EvictWeightsTaskImpl(EvictWeightsActionDummy* evict_weights);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm);
		void cancel();
	};

	ClockworkRuntimeDummy* runtime;
	std::shared_ptr<workerapi::EvictWeights> action;
	EvictWeightsTaskImpl* task;

public:
	EvictWeightsActionDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::EvictWeights> action);
	~EvictWeightsActionDummy();

	void submit();
	void handle_error(TaskError &error);

	virtual void success(std::shared_ptr<workerapi::EvictWeightsResult> result) = 0;
	virtual void error(std::shared_ptr<workerapi::ErrorResult> result) = 0;
};


class InferActionDummy {
public:

	class CopyInputTaskImpl : public CopyInputTask {
	public:
		InferActionDummy* infer;

		CopyInputTaskImpl(InferActionDummy* infer);

		void run(cudaStream_t stream);
		void process_completion();
		void success(RuntimeModel* rm, char* io_memory);
		void cancel();
	};

	class ExecTaskImpl : public ExecTask {
	public:
		InferActionDummy* infer;
		unsigned gpu_clock_before;

		ExecTaskImpl(InferActionDummy* infer);

		void run(cudaStream_t stream);
		void process_completion();
		void success();
		void cancel();
	};

	class CopyOutputTaskImpl : public CopyOutputTask {
	public:
		InferActionDummy* infer;

		CopyOutputTaskImpl(InferActionDummy* infer);

		void run(cudaStream_t stream);
		void process_completion();
		void success(char* output);
		void cancel();
	};

	ClockworkRuntimeDummy* runtime;

	std::shared_ptr<workerapi::Infer> action;

	RuntimeModel* rm;
	char* io_memory;

	bool zero_size = false;

	CopyInputTaskImpl* copy_input = nullptr;
	ExecTaskImpl* exec = nullptr;
	CopyOutputTaskImpl* copy_output = nullptr;
	
	/* Task types */
	const int copyInputTask= 0;
	const int execTask = 1;
	const int copyOutputTask = 2;

	uint64_t copy_input_earliest();

public:
	InferActionDummy(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::Infer> action);
	~InferActionDummy();

	void submit();
	void handle_completion(char* output);
	void handle_error(TaskError &error);

	virtual void success(std::shared_ptr<workerapi::InferResult> result) = 0;
	virtual void error(std::shared_ptr<workerapi::ErrorResult> result) = 0;

};



}

#endif
