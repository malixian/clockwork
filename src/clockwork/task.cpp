#include "clockwork/task.h"

#include "tbb/concurrent_queue.h"
#include <tvm/runtime/cuda_common.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/action.h"

namespace clockwork {

CudaAsyncTask::CudaAsyncTask(unsigned gpu_id, CudaEventPool* event_pool) :
	AsyncTask(gpu_id),
	event_pool(event_pool),
	async_begin_submitted(false),
	async_end_submitted(false),
	async_begin_event(event_pool->get_or_create()),
	async_end_event(event_pool->get_or_create()) {
}

CudaAsyncTask::~CudaAsyncTask() {
	event_pool->release(async_begin_event);
	event_pool->release(async_end_event);
}

void CudaAsyncTask::record_async_begin(cudaStream_t stream) {
	CUDA_CALL(cudaSetDevice(gpu_id));
	cudaError_t status = cudaEventRecord(async_begin_event, stream);
	async_begin_submitted.store(true);
}

void CudaAsyncTask::record_async_end(cudaStream_t stream) {
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(cudaEventRecord(async_end_event, stream));
	async_end_submitted.store(true);
}

bool CudaAsyncTask::is_complete() {
	CUDA_CALL(cudaSetDevice(gpu_id));

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
	CUDA_CALL(cudaSetDevice(gpu_id));
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

	std::vector<model::BatchedModel*> batched_models;
	std::vector<RuntimeModel*> runtime_models;

	try {
		for (unsigned gpu_id = 0; gpu_id < manager->num_gpus; gpu_id++) {
			if (manager->models->contains(model_id, gpu_id)) {
				throw TaskError(actionErrorInvalidModelID, "LoadModelFromDiskTask specified ID that already exists");
			}

			// TODO: loadFromDisk will call cudaMallocHost; in future don't use this, and manage host memory manually
			// TODO: for now just wrap dmlc error for failing to load model, since the existence of this task is a
			//       giant hack anyway
			model::BatchedModel* batched_model;
			try {
				batched_model = model::BatchedModel::loadFromDisk(model_path, gpu_id);
				batched_models.push_back(batched_model);
				batched_model->instantiate_models_on_host();
				batched_model->instantiate_models_on_device();
			} catch (dmlc::Error &error) {
				throw TaskError(actionErrorInvalidModelPath, error.what());
			}

			RuntimeModel* runtime_model = new RuntimeModel(batched_model, gpu_id);
			runtime_models.push_back(runtime_model);
			if (!manager->models->put_if_absent(model_id, gpu_id, runtime_model)) {
				throw TaskError(actionErrorInvalidModelID, "LoadModelFromDiskTask specified ID that already exists");
			}
		}
	} catch (TaskError e) {
		for (unsigned i = 0; i < batched_models.size(); i++) { delete batched_models[i]; }
		for (unsigned i = 0; i < runtime_models.size(); i++) { delete runtime_models[i]; }
		throw e;
	}

	// TODO Verify the following!
	// Vector runtime_models contains a separate runtime_model for each GPU ID.
	// Thus, the batched_model associated with each runtime_model is also GPU-specific.
	// For each batched_model, the parameters that are returned back to the controller,
	// i.e., input_size(1), output_size(1), implemented_batch_sizes() are identical.
	// Hence, it suffices to consider only one batched_model, ...
	// i.e., runtime_models[0]->model, when filling the LoadModelFromDiskResult object.
	// Therefore, I return only the first runtime_model to function ...
	// LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::success
	this->success(runtime_models[0]);
}


LoadWeightsTask::LoadWeightsTask(MemoryManager* manager, int model_id,
	uint64_t earliest, uint64_t latest, unsigned gpu_id,
	CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), manager(manager), model_id(model_id),
		earliest(earliest), latest(latest), rm(nullptr), new_weights(nullptr) {
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

	rm = manager->models->get(model_id, gpu_id);
	if (rm == nullptr) {
		std::string error_message = "LoadWeightsTask could not find model";
		error_message += " with model ID " + std::to_string(model_id);
		error_message += " and GPU ID " + std::to_string(gpu_id);
		throw TaskError(actionErrorUnknownModel, error_message);
	}

	rm->lock();

	this->new_version = ++rm->version;
	std::shared_ptr<Allocation> previous_weights = rm->weights;
	rm->weights = nullptr;

	rm->unlock();

	if (previous_weights != nullptr && !previous_weights->evicted) {
		manager->weights_caches[gpu_id]->unlock(previous_weights);
		manager->weights_caches[gpu_id]->free(previous_weights);
	}

	unsigned num_pages = rm->model->num_weights_pages(manager->weights_caches[gpu_id]->page_size);
	this->new_weights = manager->weights_caches[gpu_id]->alloc(num_pages, []{});
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

EvictWeightsTask::EvictWeightsTask(MemoryManager* manager, int model_id,
	uint64_t earliest, uint64_t latest, unsigned gpu_id):
		Task(gpu_id),
		manager(manager),
		model_id(model_id),
		earliest(earliest),
		latest(latest) {
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

	rm = manager->models->get(model_id, gpu_id);
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

	manager->weights_caches[gpu_id]->unlock(previous_weights);
	manager->weights_caches[gpu_id]->free(previous_weights);

	success(rm);
}


CopyInputTask::CopyInputTask(MemoryManager* manager, int model_id,
	uint64_t earliest, uint64_t latest, unsigned batch_size, size_t input_size,
	char* input, unsigned gpu_id, CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), manager(manager), model_id(model_id),
		earliest(earliest), latest(latest), batch_size(batch_size),
		input_size(input_size), input(input), rm(nullptr), io_memory(nullptr) {
}

CopyInputTask::~CopyInputTask() {
	io_memory = nullptr;
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

	rm = manager->models->get(model_id, gpu_id);
	if (rm == nullptr) {
		throw TaskError(actionErrorUnknownModel, "CopyInputTask could not find model with specified id");
	}

	if (!rm->model->is_valid_batch_size(batch_size)) {
		std::stringstream err;
		err << "CopyInputTask received unsupported batch size " << batch_size;
		throw TaskError(actionErrorInvalidBatchSize, err.str());
	}

	if (rm->model->input_size(batch_size) != input_size) {
		std::stringstream err;
		err << "CopyInputTask received incorrectly sized input"
		    << " (expected " << rm->model->input_size(batch_size) 
		    << ", got " << input_size
		    << " (batch_size=" << batch_size << ")";
		throw TaskError(actionErrorInvalidInput, err.str());		
	}

	size_t io_memory_size = rm->model->io_memory_size(batch_size);
	this->io_memory = manager->io_pools[gpu_id]->alloc(io_memory_size);

	if (this->io_memory == nullptr) {
		throw TaskError(actionErrorRuntimeError, "CopyInputTask failed to allocate memory from io_pool");
	}

	this->record_async_begin(stream);
	rm->model->transfer_input_to_device(batch_size, input, io_memory, stream);
	this->record_async_end(stream);
}

void CopyInputTask::process_completion() {
	telemetry->async_duration = this->async_duration();
	this->success(rm, io_memory);
}



ExecTask::ExecTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest,
	uint64_t latest, unsigned batch_size, char* io_memory, unsigned gpu_id,
	CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), rm(rm), manager(manager),
		earliest(earliest), latest(latest), batch_size(batch_size),
		io_memory(io_memory), weights(nullptr) {
}

ExecTask::~ExecTask() {
	weights = nullptr;

	if (workspace_memory != nullptr) {
		manager->workspace_pools[gpu_id]->free(workspace_memory);
		workspace_memory = nullptr;
	}
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

	size_t workspace_size = rm->model->workspace_memory_size(batch_size);
	this->workspace_memory = manager->workspace_pools[gpu_id]->alloc(workspace_size);

	if (this->workspace_memory == nullptr) {
		std::cout << "Trying to alloc " << workspace_size << " from " << manager->workspace_pools[0]->size << std::endl;
		throw TaskError(actionErrorRuntimeError, "ExecTask failed to allocate memory from workspace_pool");
	}

	this->record_async_begin(stream);
	rm->model->call(batch_size, weights->page_pointers, io_memory, workspace_memory, stream);
	this->record_async_end(stream);
}

void ExecTask::process_completion() {
	telemetry->async_duration = this->async_duration();
	if (workspace_memory != nullptr) {
		manager->workspace_pools[gpu_id]->free(workspace_memory);
		workspace_memory = nullptr;
	}

	rm->lock();

	int current_weights_version = rm->version;

	rm->unlock();

	if (this->weights_version != current_weights_version || weights->evicted) {
		throw TaskError(actionErrorWeightsChanged, "ExecTask failed due to weights version mismatch");
	}

	this->success();
}



CopyOutputTask::CopyOutputTask(RuntimeModel* rm, MemoryManager* manager,
	uint64_t earliest, uint64_t latest, unsigned batch_size, char* io_memory,
	unsigned gpu_id, CudaEventPool* event_pool):
		CudaAsyncTask(gpu_id, event_pool), rm(rm), manager(manager),
		earliest(earliest), latest(latest), batch_size(batch_size),
		io_memory(io_memory), output(nullptr) {
}

CopyOutputTask::~CopyOutputTask() {
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

	// TODO: this should probably be preallocated; seems silly to fail here
	size_t output_size = rm->model->output_size(batch_size);
	this->output = manager->host_io_pool->alloc(output_size);
	if (this->output == nullptr) {
		throw TaskError(actionErrorRuntimeError, "CopyOutputTask failed to allocate memory from host_io_pool");
	}

	this->record_async_begin(stream);
	rm->model->transfer_output_from_device(batch_size, output, io_memory, stream);
	this->record_async_end(stream);
}

void CopyOutputTask::process_completion() {
	telemetry->async_duration = this->async_duration();
	this->success(output);
}

}
