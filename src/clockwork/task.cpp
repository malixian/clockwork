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


CopyOutputTask::CopyOutputTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, char* output, std::shared_ptr<Allocation> workspace) : 
		Task(), AsyncTask(),
		rm(rm), cache(cache), earliest(earliest), latest(latest), output(output), workspace(workspace) {
	telemetry = new TaskTelemetry();
	copy_output_begin = event_pool.get_or_create();
	copy_output_end = event_pool.get_or_create();
}

CopyOutputTask::~CopyOutputTask() {
	event_pool.release(copy_output_begin);
	event_pool.release(copy_output_end);
	workspace = nullptr;
}

uint64_t CopyOutputTask::eligible() {
	return earliest;
}

void CopyOutputTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		error(actionErrorRuntimeError, "CopyInputTask ran before it was eligible");
		cache->unlock(workspace);
		cache->free(workspace);
		return;
	}

	if (now > latest) {
		error(actionErrorCouldNotStartInTime, "CopyInputTask could not start in time");
		cache->unlock(workspace);
		cache->free(workspace);
		return;
	}

	CUDA_CALL(cudaEventRecord(copy_output_begin, stream));
	rm->model->transfer_output_from_device(output, workspace->page_pointers, stream);
	CUDA_CALL(cudaEventRecord(copy_output_end, stream));
}

bool CopyOutputTask::is_complete() {
	cudaError_t status = cudaEventQuery(copy_output_end);
	if (status == cudaErrorNotReady) {
		return false;
	}
	CHECK(status == cudaSuccess || 
		  status == cudaErrorNotReady ||
		  status == cudaErrorCudartUnloading
		 ) << "CUDA: " << cudaGetErrorString(status);
	return true;

}

void CopyOutputTask::process_completion() {
	telemetry->async_complete = util::hrt();
	CUDA_CALL(cudaEventElapsedTime(&telemetry->async_duration, copy_output_begin, copy_output_end));

	cache->unlock(workspace);
	cache->free(workspace);

	this->success();
}


InferTask::InferTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace) : 
		Task(), AsyncTask(),
		rm(rm), cache(cache), earliest(earliest), latest(latest), workspace(workspace),
		weights(nullptr) {
	telemetry = new TaskTelemetry();
	infer_begin = event_pool.get_or_create();
	infer_end = event_pool.get_or_create();
}

InferTask::~InferTask() {
	event_pool.release(infer_begin);
	event_pool.release(infer_end);
	weights = nullptr;
	workspace = nullptr;
}

uint64_t InferTask::eligible() {
	return earliest;
}

void InferTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		error(actionErrorRuntimeError, "InferTask ran before it was eligible");
		cache->unlock(workspace);
		cache->free(workspace);
		return;
	}

	if (now > latest) {
		error(actionErrorCouldNotStartInTime, "InferTask could not start in time");
		cache->unlock(workspace);
		cache->free(workspace);
		return;
	}

	rm->lock();

	this->weights_version = rm->version;
	this->weights = rm->weights;

	rm->unlock();

	if (weights == nullptr || weights->evicted) {
		error(actionErrorModelWeightsNotPresent, "InferTask failed due to missing model weights");
		cache->unlock(workspace);
		cache->free(workspace);
		return;
	}

	CUDA_CALL(cudaEventRecord(infer_begin, stream));
	rm->model->call(weights->page_pointers, workspace->page_pointers, stream);
	CUDA_CALL(cudaEventRecord(infer_end, stream));
}

bool InferTask::is_complete() {
	cudaError_t status = cudaEventQuery(infer_end);
	if (status == cudaErrorNotReady) {
		return false;
	}
	CHECK(status == cudaSuccess || 
		  status == cudaErrorNotReady ||
		  status == cudaErrorCudartUnloading
		 ) << "CUDA: " << cudaGetErrorString(status);
	return true;

}

void InferTask::process_completion() {
	telemetry->async_complete = util::hrt();
	CUDA_CALL(cudaEventElapsedTime(&telemetry->async_duration, infer_begin, infer_end));

	rm->lock();

	int current_weights_version = rm->version;

	rm->unlock();

	if (this->weights_version != current_weights_version || weights->evicted) {
		error(actionErrorWeightsChanged, "InferTask failed due to weights version mismatch");
		cache->unlock(workspace);
		cache->free(workspace);
		return;
	}

	this->success(workspace);
}


CopyInputTask::CopyInputTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, char* input) : 
		Task(), AsyncTask(),
		rm(rm), cache(cache), earliest(earliest), latest(latest), input(input),
		workspace(nullptr) {
	telemetry = new TaskTelemetry();
	copy_input_begin = event_pool.get_or_create();
	copy_input_end = event_pool.get_or_create();
}

CopyInputTask::~CopyInputTask() {
	event_pool.release(copy_input_begin);
	event_pool.release(copy_input_end);
	workspace = nullptr;
}

uint64_t CopyInputTask::eligible() {
	return earliest;
}

void CopyInputTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		error(actionErrorRuntimeError, "CopyInputTask ran before it was eligible");
		return;
	}

	if (now > latest) {
		error(actionErrorCouldNotStartInTime, "CopyInputTask could not start in time");
		return;
	}

	this->workspace = cache->alloc(rm->model->num_workspace_pages(cache->page_size), []{});

	if (this->workspace == nullptr) {
		error(actionErrorRuntimeError, "CopyInputTask failed to allocate workspace pages from cache");
		return;
	}

	CUDA_CALL(cudaEventRecord(copy_input_begin, stream));
	rm->model->transfer_input_to_device(input, workspace->page_pointers, stream);
	CUDA_CALL(cudaEventRecord(copy_input_end, stream));
}

bool CopyInputTask::is_complete() {
	cudaError_t status = cudaEventQuery(copy_input_end);
	if (status == cudaErrorNotReady) {
		return false;
	}
	CHECK(status == cudaSuccess || 
		  status == cudaErrorNotReady ||
		  status == cudaErrorCudartUnloading
		 ) << "CUDA: " << cudaGetErrorString(status);
	return true;

}

void CopyInputTask::process_completion() {
	telemetry->async_complete = util::hrt();
	CUDA_CALL(cudaEventElapsedTime(&telemetry->async_duration, copy_input_begin, copy_input_end));

	this->success(workspace);
}


EvictWeightsTask::EvictWeightsTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest): 
		Task(), rm(rm), cache(cache), earliest(earliest), latest(latest) {
}

uint64_t EvictWeightsTask::eligible() {
	return earliest;
}

void EvictWeightsTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono, possibly use the task telemetry
	if (now < earliest) {
		error(actionErrorRuntimeError, "EvictWeightsTask ran before it was eligible");
		return;
	}

	if (now > latest) {
		error(actionErrorCouldNotStartInTime, "EvictWeightsTask could not start in time");
		return;
	}

	rm->lock();

	rm->version++;
	std::shared_ptr<Allocation> previous_weights = rm->weights;
	rm->weights = nullptr;

	rm->unlock();

	if (previous_weights == nullptr || previous_weights->evicted) {
		error(actionErrorModelWeightsNotPresent, "EvictWeightsTask not processed because no weights exist");
		return;
	}

	cache->unlock(previous_weights);
	cache->free(previous_weights);

	success();
}


LoadWeightsTask::LoadWeightsTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest) : 
		Task(), AsyncTask(),
		rm(rm), cache(cache), earliest(earliest), latest(latest), new_weights(nullptr) {
	telemetry = new TaskTelemetry();
	load_weights_begin = event_pool.get_or_create();
	load_weights_end = event_pool.get_or_create();
}

LoadWeightsTask::~LoadWeightsTask() {
	event_pool.release(load_weights_begin);
	event_pool.release(load_weights_end);
	new_weights = nullptr;
}

uint64_t LoadWeightsTask::eligible() {
	return earliest;
}

void LoadWeightsTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		error(actionErrorRuntimeError, "LoadWeightsTask ran before it was eligible");
		return;
	}

	if (now > latest) {
		error(actionErrorCouldNotStartInTime, "LoadWeightsTask could not start in time");
		return;
	}

	rm->lock();

	this->new_version = ++rm->version;
	std::shared_ptr<Allocation> previous_weights = rm->weights;
	rm->weights = nullptr;

	rm->unlock();

	if (previous_weights != nullptr && !previous_weights->evicted) {
		cache->unlock(previous_weights);
		cache->free(previous_weights);
	}

	this->new_weights = cache->alloc(rm->model->num_weights_pages(cache->page_size), []{});
	if (this->new_weights == nullptr) {
		error(actionErrorRuntimeError, "LoadWeightsTask failed to allocate pages from cache");
		return;
	}

	CUDA_CALL(cudaEventRecord(load_weights_begin, stream));
	rm->model->transfer_weights_to_device(new_weights->page_pointers, stream);
	CUDA_CALL(cudaEventRecord(load_weights_end, stream));

}

bool LoadWeightsTask::is_complete() {
	cudaError_t status = cudaEventQuery(load_weights_end);
	if (status == cudaErrorNotReady) {
		return false;
	}
	CHECK(status == cudaSuccess || 
		  status == cudaErrorNotReady ||
		  status == cudaErrorCudartUnloading
		 ) << "CUDA: " << cudaGetErrorString(status);
	return true;

}

void LoadWeightsTask::process_completion() {
	telemetry->async_complete = util::hrt();
	CUDA_CALL(cudaEventElapsedTime(&telemetry->async_duration, load_weights_begin, load_weights_end));

	bool version_unchanged = false;

	rm->lock();

	if (rm->version == this->new_version) {
		rm->version = this->new_version;
		rm->weights = this->new_weights;
		version_unchanged = true;
	}

	rm->unlock();

	if (version_unchanged) {
		success();
	} else {
		error(actionErrorWeightsInUse, "Model weights were modified while being copied");
		return;
	}
}

}