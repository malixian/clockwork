#include "clockwork/task.h"

#include "tbb/concurrent_queue.h"
#include <tvm/runtime/cuda_common.h>
#include "clockwork/api/worker_api.h"

namespace clockwork {

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


LoadModelFromDiskTask::LoadModelFromDiskTask(MemoryManager* manager, int model_id, std::string model_path, uint64_t earliest, uint64_t latest) :
		manager(manager), model_id(model_id), model_path(model_path), earliest(earliest), latest(latest) {
}

LoadModelFromDiskTask::~LoadModelFromDiskTask() {}

// Task
uint64_t LoadModelFromDiskTask::eligible() {
	return earliest;
}

void LoadModelFromDiskTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		throw TaskError(actionErrorRuntimeError, "LoadModelFromDiskTask ran before it was eligible");
	}

	if (now > latest) {
		throw TaskError(actionErrorCouldNotStartInTime, "LoadModelFromDiskTask could not start in time");
	}

	if (manager->models->contains(model_id)) {
		throw TaskError(actionErrorInvalidModelID, "LoadModelFromDiskTask specified ID that already exists");
	}

	// TODO: loadFromDisk will call cudaMallocHost; in future don't use this, and manage host memory manually
	// TODO: for now just wrap dmlc error for failing to load model, since the existence of this task is a
	//       giant hack anyway
	model::Model* model;
	try {
		model = model::Model::loadFromDisk(
			model_path + ".so",
			model_path + ".clockwork",
			model_path + ".clockwork_params"
		);
	} catch (dmlc::Error &error) {
		throw TaskError(actionErrorInvalidModelPath, error.what());
	}

	model->instantiate_model_on_host();
	model->instantiate_model_on_device();

	RuntimeModel* rm = new RuntimeModel(model);

	bool success = manager->models->put_if_absent(model_id, rm);

	if (!success) {
		delete model;
		delete rm;
		throw TaskError(actionErrorInvalidModelID, "LoadModelFromDiskTask specified ID that already exists");
	}

	this->success(rm);
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
		throw TaskError(actionErrorRuntimeError, "LoadWeightsTask ran before it was eligible");
	}

	if (now > latest) {
		throw TaskError(actionErrorCouldNotStartInTime, "LoadWeightsTask could not start in time");
	}

	rm = manager->models->get(model_id);
	if (rm == nullptr) {
		throw TaskError(actionErrorUnknownModel, "LoadWeightsTask could not find model with specified id");
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
		throw TaskError(actionErrorRuntimeError, "LoadWeightsTask failed to allocate pages from cache");
	}

	this->record_async_begin(stream);
	rm->model->transfer_weights_to_device(new_weights->page_pointers, stream);
	this->record_async_end(stream);
}

void LoadWeightsTask::process_completion() {
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
		throw TaskError(actionErrorWeightsInUse, "Model weights were modified while being copied");
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
		throw TaskError(actionErrorRuntimeError, "EvictWeightsTask ran before it was eligible");
	}

	if (now > latest) {
		throw TaskError(actionErrorCouldNotStartInTime, "EvictWeightsTask could not start in time");
	}

	rm = manager->models->get(model_id);
	if (rm == nullptr) {
		throw TaskError(actionErrorUnknownModel, "EvictWeightsTask could not find model with specified id");
	}

	rm->lock();

	rm->version++;
	std::shared_ptr<Allocation> previous_weights = rm->weights;
	rm->weights = nullptr;

	rm->unlock();

	if (previous_weights == nullptr || previous_weights->evicted) {
		throw TaskError(actionErrorModelWeightsNotPresent, "EvictWeightsTask not processed because no weights exist");
	}

	manager->weights_cache->unlock(previous_weights);
	manager->weights_cache->free(previous_weights);

	success(rm);
}


CopyInputTask::CopyInputTask(MemoryManager* manager, int model_id, uint64_t earliest, uint64_t latest, size_t input_size, char* input) : 
		manager(manager), model_id(model_id), earliest(earliest), latest(latest), input_size(input_size), input(input),
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
		throw TaskError(actionErrorRuntimeError, "CopyInputTask ran before it was eligible");
	}

	if (now > latest) {
		throw TaskError(actionErrorCouldNotStartInTime, "CopyInputTask could not start in time");
	}

	rm = manager->models->get(model_id);
	if (rm == nullptr) {
		throw TaskError(actionErrorUnknownModel, "CopyInputTask could not find model with specified id");
	}

	if (rm->model->input_size() != input_size) {
		throw TaskError(actionErrorInvalidInput, "CopyInputTask received incorrectly sized input");		
	}

	unsigned num_pages = rm->model->num_workspace_pages(manager->workspace_cache->page_size);
	this->workspace = manager->workspace_cache->alloc(num_pages, []{});

	if (this->workspace == nullptr) {
		throw TaskError(actionErrorRuntimeError, "CopyInputTask failed to allocate workspace pages from cache");
	}

	this->record_async_begin(stream);
	rm->model->transfer_input_to_device(input, workspace->page_pointers, stream);
	this->record_async_end(stream);
}

void CopyInputTask::process_completion() {
	telemetry->async_duration = this->async_duration();
	this->success(rm, workspace);
}



ExecTask::ExecTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace) : 
		rm(rm), manager(manager), earliest(earliest), latest(latest), workspace(workspace),
		weights(nullptr) {
}

ExecTask::~ExecTask() {
	weights = nullptr;
	workspace = nullptr;
}

uint64_t ExecTask::eligible() {
	return earliest;
}

void ExecTask::run(cudaStream_t stream) {
	uint64_t now = util::now(); // TODO: use chrono
	if (now < earliest) {
		throw TaskError(actionErrorRuntimeError, "ExecTask ran before it was eligible");
	}

	if (now > latest) {
		throw TaskError(actionErrorCouldNotStartInTime, "ExecTask could not start in time");
	}

	rm->lock();

	this->weights_version = rm->version;
	this->weights = rm->weights;

	rm->unlock();

	if (weights == nullptr || weights->evicted) {
		throw TaskError(actionErrorModelWeightsNotPresent, "ExecTask failed due to missing model weights");
	}

	this->record_async_begin(stream);
	rm->model->call(weights->page_pointers, workspace->page_pointers, stream);
	this->record_async_end(stream);
}

void ExecTask::process_completion() {
	telemetry->async_duration = this->async_duration();

	rm->lock();

	int current_weights_version = rm->version;

	rm->unlock();

	if (this->weights_version != current_weights_version || weights->evicted) {
		throw TaskError(actionErrorWeightsChanged, "ExecTask failed due to weights version mismatch");
	}

	this->success();
}



CopyOutputTask::CopyOutputTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace) : 
		rm(rm), manager(manager), earliest(earliest), latest(latest), workspace(workspace), output(nullptr) {
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
		throw TaskError(actionErrorRuntimeError, "CopyOutputTask ran before it was eligible");
	}

	if (now > latest) {
		throw TaskError(actionErrorCouldNotStartInTime, "CopyOutputTask could not start in time");
	}

	// TODO: use cudaHostMalloc managed host memory w/ paging
	if (!manager->io_cache->take(output)) {
		throw TaskError(actionErrorRuntimeError, "CopyOutputTask failed to allocate host pages for output");
	}

	this->record_async_begin(stream);
	rm->model->transfer_output_from_device(output, workspace->page_pointers, stream);
	this->record_async_end(stream);
}

void CopyOutputTask::process_completion() {
	telemetry->async_duration = this->async_duration();

	this->success(output);
}

}