#include "clockwork/action.h"

namespace clockwork {

LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::LoadModelFromDiskTaskImpl(LoadModelFromDiskAction* action) : LoadModelFromDiskTask(
		action->runtime->manager, 
		action->model_id,
		action->model_path,
		action->earliest, 
		action->latest), action(action) {
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::run(cudaStream_t stream) {
	LoadModelFromDiskTask::run(stream);
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::success(RuntimeModel* rm) {
	action->success();
}

void LoadModelFromDiskAction::LoadModelFromDiskTaskImpl::error(int status_code, std::string message) {
	action->error(status_code, message);
}

LoadModelFromDiskAction::LoadModelFromDiskAction(ClockworkRuntime* runtime, int model_id, std::string model_path, uint64_t earliest, uint64_t latest) :
	runtime(runtime), model_id(model_id), model_path(model_path), earliest(earliest), latest(latest), task(nullptr) {
}

LoadModelFromDiskAction::~LoadModelFromDiskAction() {
	if (task != nullptr) delete task;
}

void LoadModelFromDiskAction::submit() {
	task = new LoadModelFromDiskTaskImpl(this);
	runtime->weights_executor->enqueue(task);
};


LoadWeightsAction::LoadWeightsTaskImpl::LoadWeightsTaskImpl(LoadWeightsAction* action) : LoadWeightsTask(
		action->runtime->manager, 
		action->model_id, 
		action->earliest, 
		action->latest), action(action) {
}

void LoadWeightsAction::LoadWeightsTaskImpl::run(cudaStream_t stream) {
	LoadWeightsTask::run(stream);
	if (!has_error) action->runtime->checker->enqueue(this);
}

void LoadWeightsAction::LoadWeightsTaskImpl::success(RuntimeModel* rm) {
	action->success();
}

void LoadWeightsAction::LoadWeightsTaskImpl::error(int status_code, std::string message) {
	action->error(status_code, message);
}

LoadWeightsAction::LoadWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest) :
	runtime(runtime), model_id(model_id), earliest(earliest), latest(latest), task(nullptr) {
}

LoadWeightsAction::~LoadWeightsAction() {
	if (task != nullptr) delete task;
}

void LoadWeightsAction::submit() {
	task = new LoadWeightsTaskImpl(this);
	runtime->weights_executor->enqueue(task);
};



EvictWeightsAction::EvictWeightsTaskImpl::EvictWeightsTaskImpl(EvictWeightsAction* action) : EvictWeightsTask(
		action->runtime->manager, 
		action->model_id, 
		action->earliest, 
		action->latest), action(action) {
}

void EvictWeightsAction::EvictWeightsTaskImpl::run(cudaStream_t stream) {
	EvictWeightsTask::run(stream);
}

void EvictWeightsAction::EvictWeightsTaskImpl::success(RuntimeModel* rm) {
	action->success();
}

void EvictWeightsAction::EvictWeightsTaskImpl::error(int status_code, std::string message) {
	action->error(status_code, message);
}

EvictWeightsAction::EvictWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest) :
	runtime(runtime), model_id(model_id), earliest(earliest), latest(latest), task(nullptr) {
}

EvictWeightsAction::~EvictWeightsAction() {
	if (task != nullptr) delete task;
}

void EvictWeightsAction::submit() {
	task = new EvictWeightsTaskImpl(this);
	// Rather than have an entire new executor for this, just for now
	// use the outputs executor because it's never even close to full utilization
	runtime->outputs_executor->enqueue(task);
};


const uint64_t copy_input_lead_in = 1000000; // Copy inputs up to 1 ms before exec
uint64_t InferAction::copy_input_earliest() {
	return copy_input_lead_in > earliest ? 0 : earliest - copy_input_lead_in;
}

InferAction::CopyInputTaskImpl::CopyInputTaskImpl(InferAction* action) : CopyInputTask(
		action->runtime->manager, 
		action->model_id,
		action->copy_input_earliest(),
		action->latest,
		action->input), action(action) {
}

void InferAction::CopyInputTaskImpl::run(cudaStream_t stream) {
	CopyInputTask::run(stream);
	if (!has_error) action->runtime->checker->enqueue(this);
}

void InferAction::CopyInputTaskImpl::success(RuntimeModel* rm, std::shared_ptr<Allocation> workspace) {
	action->rm = rm;
	action->workspace = workspace;
	action->infer = new InferTaskImpl(action);
	action->runtime->gpu_executor->enqueue(action->infer);
}

void InferAction::CopyInputTaskImpl::error(int status_code, std::string message) {
	action->error(status_code, message);
}

InferAction::InferTaskImpl::InferTaskImpl(InferAction* action) : InferTask(
		action->rm, 
		action->runtime->manager, 
		action->earliest, 
		action->latest, 
		action->workspace), action(action) {
}

void InferAction::InferTaskImpl::run(cudaStream_t stream) {
	InferTask::run(stream);
	if (!has_error) action->runtime->checker->enqueue(this);
}

void InferAction::InferTaskImpl::success() {
	action->copy_output = new CopyOutputTaskImpl(action);
	action->runtime->outputs_executor->enqueue(action->copy_output);
}

void InferAction::InferTaskImpl::error(int status_code, std::string message) {
	action->error(status_code, message);
}

InferAction::CopyOutputTaskImpl::CopyOutputTaskImpl(InferAction* action) : CopyOutputTask(
		action->rm,
		action->runtime->manager,
		0,
		18446744073709551615UL,
		action->output,
		action->workspace), action(action) {
}

void InferAction::CopyOutputTaskImpl::run(cudaStream_t stream) {
	CopyOutputTask::run(stream);
	if (!has_error) action->runtime->checker->enqueue(this);
}

void InferAction::CopyOutputTaskImpl::success() {
	action->success();
}

void InferAction::CopyOutputTaskImpl::error(int status_code, std::string message) {
	action->error(status_code, message);
}

InferAction::InferAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest, char* input, char* output) :
		runtime(runtime), model_id(model_id), earliest(earliest), latest(latest), input(input), output(output),
		rm(nullptr), workspace(nullptr) {
	copy_input = new CopyInputTaskImpl(this);
}

InferAction::~InferAction() {
	delete copy_input;
	if (infer != nullptr) delete infer;
	if (copy_output != nullptr) delete copy_output;
	workspace = nullptr;
}

void InferAction::submit() {
	runtime->inputs_executor->enqueue(copy_input);
};

}