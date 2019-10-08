#ifndef _CLOCKWORK_TASK_H_
#define _CLOCKWORK_TASK_H_

#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/telemetry.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"

namespace clockwork {


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

class ModelStore {
private:
	std::atomic_flag in_use;
	std::unordered_map<int, RuntimeModel*> models;

public:

	ModelStore() : in_use(ATOMIC_FLAG_INIT) {}

	RuntimeModel* get(int model_id) {
		while (in_use.test_and_set());

		RuntimeModel* rm = models[model_id];

		in_use.clear();

		return rm;
	}

	void put(int model_id, RuntimeModel* model) {
		while (in_use.test_and_set());

		models[model_id] = model;

		in_use.clear();
	}

};

class MemoryManager {
public:
	PageCache* weights_cache;
	PageCache* workspace_cache;
	ModelStore* models;
};

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


class CudaAsyncTask : public AsyncTask {
private:
	std::atomic_bool async_begin_submitted, async_end_submitted;
	cudaEvent_t async_begin_event, async_end_event;
public:
	CudaAsyncTask();
	~CudaAsyncTask();

	void record_async_begin(cudaStream_t stream);
	void record_async_end(cudaStream_t stream);
	float async_duration();

	// AsyncTask
	bool is_complete();
	virtual void process_completion() = 0;
};

class WithError {
public:
	std::atomic_bool has_error;

	WithError() : has_error(false) {}

	// Callback
	virtual void error(int status_code, std::string message) = 0;


	void set_error(int status_code, std::string message) {
		has_error.store(true);
		this->error(status_code, message);
	}
};

class LoadWeightsTask : public Task, public CudaAsyncTask, public WithError {
private:
	int model_id;
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;

	int new_version;
	std::shared_ptr<Allocation> new_weights;

public:

	LoadWeightsTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest);
	~LoadWeightsTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;
	virtual void error(int status_code, std::string message) = 0;

};

class EvictWeightsTask : public Task, public WithError {
private:
	int model_id;
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;

public:

	EvictWeightsTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest);

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;
	virtual void error(int status_code, std::string message) = 0;

};

class CopyInputTask : public Task, public CudaAsyncTask, public WithError {
private:
	MemoryManager* manager;

	int model_id;
	uint64_t earliest, latest;
	char* input;

	RuntimeModel* rm;
	std::shared_ptr<Allocation> workspace;

public:

	CopyInputTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest, char* input);
	~CopyInputTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success(RuntimeModel* rm, std::shared_ptr<Allocation> workspace) = 0;
	virtual void error(int status_code, std::string message) = 0;
};

class InferTask : public Task, public CudaAsyncTask, public WithError {
private:
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;
	char* output;

	int weights_version;
	std::shared_ptr<Allocation> weights;
	std::shared_ptr<Allocation> workspace;

public:

	InferTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace);
	~InferTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void error(int status_code, std::string message) = 0;
	virtual void success() = 0;
};

class CopyOutputTask : public Task, public CudaAsyncTask, public WithError {
private:
	RuntimeModel* rm;
	MemoryManager* manager;

	uint64_t earliest, latest;
	char* output;

	std::shared_ptr<Allocation> workspace;

public:
	CopyOutputTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, char* output, std::shared_ptr<Allocation> workspace);
	~CopyOutputTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void error(int status_code, std::string message) = 0;
	virtual void success() = 0;
};

}

#endif
