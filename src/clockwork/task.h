#ifndef _CLOCKWORK_TASK_H_
#define _CLOCKWORK_TASK_H_

#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/telemetry.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"

/*
This file contains logic for executing models directly
*/

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

	~ModelStore() {
		while (in_use.test_and_set());

		for (auto &p : models) {
			RuntimeModel* rm = p.second;
			if (rm != nullptr) {
				// TODO: models are definitely not always on device or host
				rm->model->uninstantiate_model_on_device();
				rm->model->uninstantiate_model_on_host();
				delete rm->model;
				delete rm;
			}
		}

		// Let callers hang here to aid in use-after-free
		// in_use.clear();
	}

	RuntimeModel* get(int model_id) {
		while (in_use.test_and_set());

		RuntimeModel* rm = models[model_id];

		in_use.clear();

		return rm;
	}

	bool contains(int model_id) {
		while (in_use.test_and_set());

		RuntimeModel* rm = models[model_id];

		in_use.clear();

		return rm != nullptr;
	}

	void put(int model_id, RuntimeModel* model) {
		while (in_use.test_and_set());

		models[model_id] = model;

		in_use.clear();
	}

	bool put_if_absent(int model_id, RuntimeModel* model) {
		while (in_use.test_and_set());

		bool did_put = false;
		if (models[model_id] == nullptr) {
			models[model_id] = model;
			did_put = true;
		}

		in_use.clear();

		return did_put;
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
	std::shared_ptr<TaskTelemetry> telemetry;

	Task() : telemetry(std::make_shared<TaskTelemetry>()) {}

	virtual uint64_t eligible() = 0;
	virtual void run(cudaStream_t stream) = 0;
	virtual void cancel() = 0;
};

class AsyncTask : public Task {
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

class TaskError {
public:
	int status_code;
	std::string message;
	TaskError(int status_code, std::string message) : status_code(status_code), message(message) {}
};

/* For now, load, deserialize, and instantiate on host and device all in one.  TODO: split up. */
class LoadModelFromDiskTask : public Task {
private:
	int model_id;
	std::string model_path;

	MemoryManager* manager;
	uint64_t earliest, latest;

public:

	LoadModelFromDiskTask(MemoryManager* manager, int model_id, std::string model_path, uint64_t earliest, uint64_t latest);
	~LoadModelFromDiskTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);
	virtual void cancel() = 0;

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;

};

class LoadWeightsTask : public CudaAsyncTask {
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
	virtual void cancel() = 0;

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;

};

class EvictWeightsTask : public Task {
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
	virtual void cancel() = 0;

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;

};

class CopyInputTask : public CudaAsyncTask {
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
	virtual void cancel() = 0;

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success(RuntimeModel* rm, std::shared_ptr<Allocation> workspace) = 0;
};

class ExecTask : public CudaAsyncTask {
private:
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;

	int weights_version;
	std::shared_ptr<Allocation> weights;
	std::shared_ptr<Allocation> workspace;

public:

	ExecTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace);
	~ExecTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);
	virtual void cancel() = 0;

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success() = 0;
};

class CopyOutputTask : public CudaAsyncTask {
private:
	RuntimeModel* rm;
	MemoryManager* manager;

	uint64_t earliest, latest;
	char* output;

	std::shared_ptr<Allocation> workspace;

public:
	CopyOutputTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace);
	~CopyOutputTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);
	virtual void cancel() = 0;

	// CudaAsyncTask
	void process_completion();

	// Callbacks
	virtual void success(char* output) = 0;
};

}

#endif
