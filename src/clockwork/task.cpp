#include "clockwork/task.h"

#include "tbb/concurrent_queue.h"
#include <tvm/runtime/cuda_common.h>
#include "clockwork/api/worker_api.h"

namespace clockwork {


RuntimeModel::RuntimeModel(model::Model* model) : model(model), in_use(ATOMIC_FLAG_INIT), weights(nullptr), version(0) {
}

bool RuntimeModel::try_lock() {
	return !in_use.test_and_set();
}

void RuntimeModel::lock() {
	while (!try_lock());
}

void RuntimeModel::unlock() {
	in_use.clear();
}


class CudaEventPool {
public:
	tbb::concurrent_queue<cudaEvent_t> events;

	cudaEvent_t get_or_create() {
		cudaEvent_t event;
		if (!events.try_pop(event)) {
			CUDA_CALL(cudaEventCreate(&event));
		}
		return event;
	}

	void release(cudaEvent_t event) {
		events.push(event);
	}

};

CudaEventPool event_pool;

CudaAsyncTask::CudaAsyncTask() :
		AsyncTask(),
		async_begin_submitted(false), 
		async_end_submitted(false), 
		async_begin_event(event_pool.get_or_create()), 
		async_end_event(event_pool.get_or_create()) {
}

CudaAsyncTask::~CudaAsyncTask() {
	event_pool.release(async_begin_event);
	event_pool.release(async_end_event);
}

void CudaAsyncTask::record_async_begin(cudaStream_t stream) {
	CUDA_CALL(cudaEventRecord(async_begin_event, stream));
	async_begin_submitted.store(true);
}

void CudaAsyncTask::record_async_end(cudaStream_t stream) {
	CUDA_CALL(cudaEventRecord(async_end_event, stream));
	async_end_submitted.store(true);
}

bool CudaAsyncTask::is_complete() {
	// Same semantics as cuda event: unused event is complete
	if (!async_begin_submitted.load()) return true;
	if (!async_end_submitted.load()) return false;

	cudaError_t status = cudaEventQuery(async_end_event);
	if (status == cudaErrorNotReady) {
		return false;
	}
	CHECK(status == cudaSuccess || 
		  status == cudaErrorNotReady ||
		  status == cudaErrorCudartUnloading
		 ) << "CUDA: " << cudaGetErrorString(status);
	return true;
}

float CudaAsyncTask::async_duration() {
	float async_duration;
	CUDA_CALL(cudaEventElapsedTime(&async_duration, async_begin_event, async_end_event));
	return async_duration;
}


LoadWeightsTask::LoadWeightsTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest) : 
		manager(manager), model_id(model_id), earliest(earliest), latest(latest), 
		rm(nullptr), new_weights(nullptr) {
}

LoadWeightsTask::~LoadWeightsTask() {
	new_weights = nullptr;
}

uint64_t LoadWeightsTask::eligible() {
	return earliest;
}

void LoadWeightsTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		set_error(actionErrorRuntimeError, "LoadWeightsTask ran before it was eligible");
		return;
	}

	if (now > latest) {
		set_error(actionErrorCouldNotStartInTime, "LoadWeightsTask could not start in time");
		return;
	}

	rm = manager->models->get(model_id);
	if (rm == nullptr) {
		set_error(actionErrorUnknownModel, "LoadWeightsTask could not find model with specified id");
		return;
	}

	rm->lock();

	this->new_version = ++rm->version;
	std::shared_ptr<Allocation> previous_weights = rm->weights;
	rm->weights = nullptr;

	rm->unlock();

	if (previous_weights != nullptr && !previous_weights->evicted) {
		manager->weights_cache->unlock(previous_weights);
		manager->weights_cache->free(previous_weights);
	}

	unsigned num_pages = rm->model->num_weights_pages(manager->weights_cache->page_size);
	this->new_weights = manager->weights_cache->alloc(num_pages, []{});
	if (this->new_weights == nullptr) {
		set_error(actionErrorRuntimeError, "LoadWeightsTask failed to allocate pages from cache");
		return;
	}

	this->record_async_begin(stream);
	rm->model->transfer_weights_to_device(new_weights->page_pointers, stream);
	this->record_async_end(stream);
}

void LoadWeightsTask::process_completion() {
	telemetry->async_complete = util::hrt();
	telemetry->async_duration = this->async_duration();

	bool version_unchanged = false;

	rm->lock();

	if (rm->version == this->new_version) {
		rm->version = this->new_version;
		rm->weights = this->new_weights;
		version_unchanged = true;
	}

	rm->unlock();

	if (version_unchanged) {
		success(rm);
	} else {
		set_error(actionErrorWeightsInUse, "Model weights were modified while being copied");
		return;
	}
}

EvictWeightsTask::EvictWeightsTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest): 
		manager(manager), model_id(model_id), earliest(earliest), latest(latest) {
}

uint64_t EvictWeightsTask::eligible() {
	return earliest;
}

void EvictWeightsTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono, possibly use the task telemetry
	if (now < earliest) {
		set_error(actionErrorRuntimeError, "EvictWeightsTask ran before it was eligible");
		return;
	}

	if (now > latest) {
		set_error(actionErrorCouldNotStartInTime, "EvictWeightsTask could not start in time");
		return;
	}

	rm = manager->models->get(model_id);
	if (rm == nullptr) {
		set_error(actionErrorUnknownModel, "EvictWeightsTask could not find model with specified id");
		return;
	}

	rm->lock();

	rm->version++;
	std::shared_ptr<Allocation> previous_weights = rm->weights;
	rm->weights = nullptr;

	rm->unlock();

	if (previous_weights == nullptr || previous_weights->evicted) {
		set_error(actionErrorModelWeightsNotPresent, "EvictWeightsTask not processed because no weights exist");
		return;
	}

	manager->weights_cache->unlock(previous_weights);
	manager->weights_cache->free(previous_weights);

	success(rm);
}


CopyInputTask::CopyInputTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest, char* input) : 
		manager(manager), model_id(model_id), earliest(earliest), latest(latest), input(input),
		rm(nullptr), workspace(nullptr) {
}

CopyInputTask::~CopyInputTask() {
	workspace = nullptr;
}

uint64_t CopyInputTask::eligible() {
	return earliest;
}

void CopyInputTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		set_error(actionErrorRuntimeError, "CopyInputTask ran before it was eligible");
		return;
	}

	if (now > latest) {
		set_error(actionErrorCouldNotStartInTime, "CopyInputTask could not start in time");
		return;
	}

	rm = manager->models->get(model_id);
	if (rm == nullptr) {
		set_error(actionErrorUnknownModel, "CopyInputTask could not find model with specified id");
		return;
	}

	unsigned num_pages = rm->model->num_workspace_pages(manager->workspace_cache->page_size);
	this->workspace = manager->workspace_cache->alloc(num_pages, []{});

	if (this->workspace == nullptr) {
		set_error(actionErrorRuntimeError, "CopyInputTask failed to allocate workspace pages from cache");
		return;
	}

	this->record_async_begin(stream);
	rm->model->transfer_input_to_device(input, workspace->page_pointers, stream);
	this->record_async_end(stream);
}

void CopyInputTask::process_completion() {
	telemetry->async_complete = util::hrt();
	telemetry->async_duration = this->async_duration();
	this->success(rm, workspace);
}



InferTask::InferTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace) : 
		rm(rm), manager(manager), earliest(earliest), latest(latest), workspace(workspace),
		weights(nullptr) {
}

InferTask::~InferTask() {
	weights = nullptr;
	workspace = nullptr;
}

uint64_t InferTask::eligible() {
	return earliest;
}

void InferTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		set_error(actionErrorRuntimeError, "InferTask ran before it was eligible");
		manager->workspace_cache->unlock(workspace);
		manager->workspace_cache->free(workspace);
		return;
	}

	if (now > latest) {
		set_error(actionErrorCouldNotStartInTime, "InferTask could not start in time");
		manager->workspace_cache->unlock(workspace);
		manager->workspace_cache->free(workspace);
		return;
	}

	rm->lock();

	this->weights_version = rm->version;
	this->weights = rm->weights;

	rm->unlock();

	if (weights == nullptr || weights->evicted) {
		set_error(actionErrorModelWeightsNotPresent, "InferTask failed due to missing model weights");
		manager->workspace_cache->unlock(workspace);
		manager->workspace_cache->free(workspace);
		return;
	}

	this->record_async_begin(stream);
	rm->model->call(weights->page_pointers, workspace->page_pointers, stream);
	this->record_async_end(stream);
}

void InferTask::process_completion() {
	telemetry->async_complete = util::hrt();
	telemetry->async_duration = this->async_duration();

	rm->lock();

	int current_weights_version = rm->version;

	rm->unlock();

	if (this->weights_version != current_weights_version || weights->evicted) {
		set_error(actionErrorWeightsChanged, "InferTask failed due to weights version mismatch");
		manager->workspace_cache->unlock(workspace);
		manager->workspace_cache->free(workspace);
		return;
	}

	this->success();
}



CopyOutputTask::CopyOutputTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, char* output, std::shared_ptr<Allocation> workspace) : 
		rm(rm), manager(manager), earliest(earliest), latest(latest), output(output), workspace(workspace) {
}

CopyOutputTask::~CopyOutputTask() {
	workspace = nullptr;
}

uint64_t CopyOutputTask::eligible() {
	return earliest;
}

void CopyOutputTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		set_error(actionErrorRuntimeError, "CopyOutputTask ran before it was eligible");
		manager->workspace_cache->unlock(workspace);
		manager->workspace_cache->free(workspace);
		return;
	}

	if (now > latest) {
		set_error(actionErrorCouldNotStartInTime, "CopyOutputTask could not start in time");
		manager->workspace_cache->unlock(workspace);
		manager->workspace_cache->free(workspace);
		return;
	}

	this->record_async_begin(stream);
	rm->model->transfer_output_from_device(output, workspace->page_pointers, stream);
	this->record_async_end(stream);
}

void CopyOutputTask::process_completion() {
	telemetry->async_complete = util::hrt();
	telemetry->async_duration = this->async_duration();

	manager->workspace_cache->unlock(workspace);
	manager->workspace_cache->free(workspace);

	this->success();
}

}