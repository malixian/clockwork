#ifndef _CLOCKWORK_TASK_H_
#define _CLOCKWORK_TASK_H_

#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/telemetry.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"

namespace clockwork {

class Task {
public:
	TaskTelemetry* telemetry;

	Task() : telemetry(new TaskTelemetry()) {}

	virtual uint64_t eligible() = 0;
	virtual void run(cudaStream_t stream) = 0;
};

class AsyncTask {
public:
	virtual bool is_complete() = 0;
	virtual void process_completion() = 0;
};


class RuntimeModel {
public:
	model::Model* model;
	std::atomic_flag in_use;
	int version;
	std::shared_ptr<Allocation> weights;

	RuntimeModel(model::Model* model);

	bool try_lock();
	void lock();
	void unlock();

};

class CopyOutputTask : public Task, public AsyncTask {
private:
	bool submitted;
	RuntimeModel* rm;
	PageCache* cache;
	uint64_t earliest, latest;
	char* output;

	std::shared_ptr<Allocation> workspace;

	cudaEvent_t copy_output_begin, copy_output_end;

public:
	CopyOutputTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, char* output, std::shared_ptr<Allocation> workspace);
	~CopyOutputTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// AsyncTask
	bool is_complete();
	void process_completion();

	// Callbacks
	virtual void error(int status_code, std::string message) = 0;
	virtual void success() = 0;
};

class InferTask : public Task, public AsyncTask {
private:
	bool submitted;
	RuntimeModel* rm;
	PageCache* cache;
	uint64_t earliest, latest;
	char* output;

	int weights_version;
	std::shared_ptr<Allocation> weights;
	std::shared_ptr<Allocation> workspace;

	cudaEvent_t infer_begin, infer_end;

public:

	InferTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace);
	~InferTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// AsyncTask
	bool is_complete();
	void process_completion();

	// Callbacks
	virtual void error(int status_code, std::string message) = 0;
	virtual void success(std::shared_ptr<Allocation> workspace) = 0;
};

class CopyInputTask : public Task, public AsyncTask {
private:
	bool submitted;
	RuntimeModel* rm;
	PageCache* cache;
	uint64_t earliest, latest;
	char* input;

	std::shared_ptr<Allocation> workspace;

	cudaEvent_t copy_input_begin, copy_input_end;

public:

	CopyInputTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, char* input);
	~CopyInputTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// AsyncTask
	bool is_complete();
	void process_completion();

	// Callbacks
	virtual void success(std::shared_ptr<Allocation> workspace) = 0;
	virtual void error(int status_code, std::string message) = 0;
};

class EvictWeightsTask : public Task {
private:
	RuntimeModel* rm;
	PageCache* cache;
	uint64_t earliest, latest;

public:

	EvictWeightsTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest);

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// Callbacks
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;

};

class LoadWeightsTask : public Task, public AsyncTask {
private:
	bool submitted;
	RuntimeModel* rm;
	PageCache* cache;
	uint64_t earliest, latest;

	int new_version;
	std::shared_ptr<Allocation> new_weights;

	cudaEvent_t load_weights_begin;
	cudaEvent_t load_weights_end;

public:

	LoadWeightsTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest);
	~LoadWeightsTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// AsyncTask
	bool is_complete();
	void process_completion();

	// Callbacks
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;

};

}

#endif
