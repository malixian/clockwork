#include "clockwork/action.h"
#include "clockwork/telemetry.h"

namespace clockwork {

void extract_timing_sync(workerapi::Timing* timing, std::shared_ptr<TaskTelemetry> &telemetry) {
	timing->begin = util::nanos(telemetry->dequeued);
	timing->end = util::now();
	timing->duration = timing->end - timing->begin;
}

void extract_timing_async(workerapi::Timing* timing, std::shared_ptr<TaskTelemetry> &telemetry) {
	timing->begin = util::nanos(telemetry->dequeued);
	timing->end = util::nanos(telemetry->async_complete);
	timing->duration = (uint64_t) (telemetry->async_duration * 1000000.0);
}

LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::LoadModelFromDiskTaskImpl(LoadModelFromDiskAction* load_model) : 
		LoadModelFromDiskTask(
			load_model->runtime->manager,
			load_model->action->model_id,
			load_model->action->model_path,
			load_model->action->earliest,
			load_model->action->latest), 
		load_model(load_model) {
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::run(cudaStream_t stream) {
	try {
		LoadModelFromDiskTask::run(stream);
	} catch (TaskError &error) {
		load_model->handle_error(error);
	}
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::success(RuntimeModel* rm) {
	auto result = std::make_shared<workerapi::LoadModelFromDiskResult>();

	result->id = load_model->action->id;
	result->action_type = workerapi::loadModelFromDiskAction;
	result->status = actionSuccess;
	result->input_size = rm->model->input_size(1);
	result->output_size = rm->model->output_size(1);
	result->supported_batch_sizes = rm->model->implemented_batch_sizes();

	int page_size = load_model->runtime->manager->weights_cache->page_size;
	result->weights_size_in_cache = rm->model->num_weights_pages(page_size) * page_size;

	extract_timing_sync(result.get(), telemetry);

	load_model->success(result);
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	load_model->handle_error(error);
}

LoadModelFromDiskAction::LoadModelFromDiskAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadModelFromDisk> action) :
	runtime(runtime), action(action), task(nullptr) {
}

LoadModelFromDiskAction::~LoadModelFromDiskAction() {
	if (task != nullptr) delete task;
	action = nullptr;
}

void LoadModelFromDiskAction::submit() {
	CHECK(task == nullptr);
	task = new LoadModelFromDiskTaskImpl(this);
	runtime->load_model_executor->enqueue(task);
}

void LoadModelFromDiskAction::handle_error(TaskError &error) {
	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = workerapi::loadModelFromDiskAction;
	result->status = error.status_code;
	result->message = error.message;

	this->error(result);
}




LoadWeightsAction::LoadWeightsTaskImpl::LoadWeightsTaskImpl(LoadWeightsAction* load_weights) : LoadWeightsTask(
			load_weights->runtime->manager, 
			load_weights->action->model_id, 
			load_weights->action->earliest,
			load_weights->action->latest),
		load_weights(load_weights) {
}

void LoadWeightsAction::LoadWeightsTaskImpl::run(cudaStream_t stream) {
	try {
		LoadWeightsTask::run(stream);
		load_weights->runtime->checker->enqueue(this);
	} catch (TaskError &error) {
		load_weights->handle_error(error);
	}
}

void LoadWeightsAction::LoadWeightsTaskImpl::process_completion() {
	try {
		LoadWeightsTask::process_completion();
	} catch (TaskError &error) {
		load_weights->handle_error(error);
	}
}

void LoadWeightsAction::LoadWeightsTaskImpl::success(RuntimeModel* rm) {
	auto result = std::make_shared<workerapi::LoadWeightsResult>();

	result->id = load_weights->action->id;
	result->action_type = workerapi::loadWeightsAction;
	result->status = actionSuccess;

	extract_timing_async(result.get(), telemetry);

	load_weights->success(result);
}

void LoadWeightsAction::LoadWeightsTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	load_weights->handle_error(error);
}

LoadWeightsAction::LoadWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadWeights> action) :
	runtime(runtime), action(action), task(nullptr) {
}

LoadWeightsAction::~LoadWeightsAction() {
	if (task != nullptr) delete task;
	action = nullptr;
}

void LoadWeightsAction::submit() {
	task = new LoadWeightsTaskImpl(this);
	runtime->weights_executor->enqueue(task);
}

void LoadWeightsAction::handle_error(TaskError &error) {
	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = workerapi::loadWeightsAction;
	result->status = error.status_code;
	result->message = error.message;

	this->error(result);
}





EvictWeightsAction::EvictWeightsTaskImpl::EvictWeightsTaskImpl(EvictWeightsAction* evict_weights) : 
		EvictWeightsTask(
			evict_weights->runtime->manager, 
			evict_weights->action->model_id, 
			evict_weights->action->earliest, 
			evict_weights->action->latest), 
		evict_weights(evict_weights) {
}

void EvictWeightsAction::EvictWeightsTaskImpl::run(cudaStream_t stream) {
	try {
		EvictWeightsTask::run(stream);
	} catch (TaskError &error) {
		evict_weights->handle_error(error);
	}
}

void EvictWeightsAction::EvictWeightsTaskImpl::success(RuntimeModel* rm) {
	auto result = std::make_shared<workerapi::EvictWeightsResult>();

	result->id = evict_weights->action->id;
	result->action_type = workerapi::evictWeightsAction;
	result->status = actionSuccess;

	extract_timing_sync(result.get(), telemetry);

	evict_weights->success(result);
}

void EvictWeightsAction::EvictWeightsTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	evict_weights->handle_error(error);
}

EvictWeightsAction::EvictWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::EvictWeights> action) :
	runtime(runtime), action(action), task(nullptr) {
}

EvictWeightsAction::~EvictWeightsAction() {
	if (task != nullptr) delete task;
	action = nullptr;
}

void EvictWeightsAction::submit() {
	task = new EvictWeightsTaskImpl(this);
	// Rather than have an entire new executor for this, just for now
	// use the outputs executor because it's never even close to full utilization
	runtime->outputs_executor->enqueue(task);
}

void EvictWeightsAction::handle_error(TaskError &error) {
	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = workerapi::evictWeightsAction;
	result->status = error.status_code;
	result->message = error.message;

	this->error(result);
}




const uint64_t copy_input_lead_in = 1000000; // Copy inputs up to 1 ms before exec
uint64_t InferAction::copy_input_earliest() {
	return copy_input_lead_in > action->earliest ? 0 : action->earliest - copy_input_lead_in;
}

InferAction::CopyInputTaskImpl::CopyInputTaskImpl(InferAction* infer) : CopyInputTask(
		infer->runtime->manager, 
		infer->action->model_id,
		infer->copy_input_earliest(),
		infer->action->latest,
		infer->action->batch_size,
		infer->action->input_size,
		infer->action->input), infer(infer) {
}

void InferAction::CopyInputTaskImpl::run(cudaStream_t stream) {
	try {
		CopyInputTask::run(stream);
		infer->runtime->checker->enqueue(this);
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::CopyInputTaskImpl::process_completion() {
	try {
		CopyInputTask::process_completion();
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::CopyInputTaskImpl::success(RuntimeModel* rm, char* io_memory) {
	infer->rm = rm;
	infer->io_memory = io_memory;
	infer->exec = new ExecTaskImpl(infer);
	infer->runtime->gpu_executor->enqueue(infer->exec);
}

void InferAction::CopyInputTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	infer->handle_error(error);
}

InferAction::ExecTaskImpl::ExecTaskImpl(InferAction* infer) : ExecTask(
		infer->rm,
		infer->runtime->manager, 
		infer->action->earliest,
		infer->action->latest, 
		infer->action->batch_size,
		infer->io_memory), infer(infer) {
}

void InferAction::ExecTaskImpl::run(cudaStream_t stream) {
	try {
		ExecTask::run(stream);
		infer->runtime->checker->enqueue(this);
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::ExecTaskImpl::process_completion() {
	try {
		ExecTask::process_completion();
	} catch (TaskError &error) {
		infer->handle_error(error);
	}	
}

void InferAction::ExecTaskImpl::success() {
	infer->copy_output = new CopyOutputTaskImpl(infer);
	infer->runtime->outputs_executor->enqueue(infer->copy_output);
}

void InferAction::ExecTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	infer->handle_error(error);
}

InferAction::CopyOutputTaskImpl::CopyOutputTaskImpl(InferAction* infer) : CopyOutputTask(
		infer->rm,
		infer->runtime->manager,
		0,
		18446744073709551615UL,
		infer->action->batch_size,
		infer->io_memory), infer(infer) {
}

void InferAction::CopyOutputTaskImpl::run(cudaStream_t stream) {
	try {
		CopyOutputTask::run(stream);
		infer->runtime->checker->enqueue(this);
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::CopyOutputTaskImpl::process_completion() {
	try {
		CopyOutputTask::process_completion();
	} catch (TaskError &error) {
		infer->handle_error(error);
	}
}

void InferAction::CopyOutputTaskImpl::success(char* output) {
	infer->handle_completion(output);
}

void InferAction::CopyOutputTaskImpl::cancel() {
	TaskError error(actionCancelled, "Action cancelled");
	infer->handle_error(error);
}

InferAction::InferAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::Infer> action) :
		runtime(runtime), action(action), rm(nullptr), io_memory(nullptr) {
}

InferAction::~InferAction() {
	if (copy_input != nullptr) delete copy_input;
	if (exec != nullptr) delete exec;
	if (copy_output != nullptr) delete copy_output;
	io_memory = nullptr;
}

void InferAction::submit() {
	copy_input = new CopyInputTaskImpl(this);
	runtime->inputs_executor->enqueue(copy_input);
}

void InferAction::handle_completion(char* output) {
	runtime->manager->io_pool->free(io_memory);
	io_memory = nullptr;

	auto result = std::make_shared<workerapi::InferResult>();

	result->id = action->id;
	result->action_type = workerapi::inferAction;
	result->status = actionSuccess;

	exec->telemetry->action_id = action->id;
	exec->telemetry->model_id = action->model_id;
	exec->telemetry->task_type = workerapi::inferAction;

	extract_timing_async(&result->copy_input, copy_input->telemetry);
	extract_timing_async(&result->exec, exec->telemetry);
	extract_timing_async(&result->copy_output, copy_output->telemetry);

	runtime->telemetry_logger->log(exec->telemetry);

	result->output_size = rm->model->output_size(action->batch_size);
	result->output = output;

	// TODO: who / where frees the output memory?
	this->success(result);
}

void InferAction::handle_error(TaskError &error) {
	if (io_memory != nullptr) {
		runtime->manager->io_pool->free(io_memory);
		io_memory = nullptr;
	}

	auto result = std::make_shared<workerapi::ErrorResult>();

	result->id = action->id;
	result->action_type = workerapi::inferAction;
	result->status = error.status_code;
	result->message = error.message;

	this->error(result);
}

}