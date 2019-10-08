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

class CopyOutputTask : public Task, public AsyncTask {
private:
	bool submitted;
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;
	char* output;

	std::shared_ptr<Allocation> workspace;

	cudaEvent_t copy_output_begin, copy_output_end;

public:
	CopyOutputTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, char* output, std::shared_ptr<Allocation> workspace);
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
	MemoryManager* manager;
	uint64_t earliest, latest;
	char* output;

	int weights_version;
	std::shared_ptr<Allocation> weights;
	std::shared_ptr<Allocation> workspace;

	cudaEvent_t infer_begin, infer_end;

public:

	InferTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace);
	~InferTask();

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

class CopyInputTask : public Task, public AsyncTask {
private:
	bool submitted;
	int model_id;
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;
	char* input;

	std::shared_ptr<Allocation> workspace;

	cudaEvent_t copy_input_begin, copy_input_end;

public:

	CopyInputTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest, char* input);
	~CopyInputTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// AsyncTask
	bool is_complete();
	void process_completion();

	// Callbacks
	virtual void success(RuntimeModel* rm, std::shared_ptr<Allocation> workspace) = 0;
	virtual void error(int status_code, std::string message) = 0;
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

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;
	virtual void error(int status_code, std::string message) = 0;

};

class LoadWeightsTask : public Task, public AsyncTask {
private:
	bool submitted;
	int model_id;
	RuntimeModel* rm;
	MemoryManager* manager;
	uint64_t earliest, latest;

	int new_version;
	std::shared_ptr<Allocation> new_weights;

	cudaEvent_t load_weights_begin;
	cudaEvent_t load_weights_end;

public:

	LoadWeightsTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest);
	~LoadWeightsTask();

	// Task
	uint64_t eligible();
	void run(cudaStream_t stream);

	// AsyncTask
	bool is_complete();
	void process_completion();

	// Callbacks
	virtual void success(RuntimeModel* rm) = 0;
	virtual void error(int status_code, std::string message) = 0;

};

}

#endif
