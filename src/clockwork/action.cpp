#include "clockwork/action.h"

namespace clockwork {

Executor::Executor(TaskType type) : alive(true), type(type){
	unsigned numThreads = 1;
	for (unsigned i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&Executor::executorMain, this, i));
	}
}

void Executor::enqueue(Task* task) {
	queue.enqueue(task, task->eligible());
}

void Executor::shutdown() {
	alive.store(false);
	queue.shutdown();
}

void Executor::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

void Executor::executorMain(int executorId) {
	// TODO: acquire core, bind to core, set thread priority
	util::initializeCudaStream();
	cudaStream_t stream = util::Stream();

	while (alive.load()) {
		// TODO: possibility off too many outstanding asyc tasks

		// TODO: queue should spin-wait rather than blocking
		// TODO: shutdown queue or use try_dequeue
		Task* next = queue.dequeue();

		next->telemetry->dequeued = util::hrt();
		next->run(stream);
		next->telemetry->exec_complete = util::hrt();
	}
}


AsyncTaskChecker::AsyncTaskChecker() : alive(true) {
	unsigned num_threads = 1;
	for (unsigned i = 0; i < num_threads; i++) {
		threads.push_back(std::thread(&AsyncTaskChecker::executorMain, this, i));
	}
}

void AsyncTaskChecker::enqueue(AsyncTask* task) {
	queue.push(task);
}

void AsyncTaskChecker::shutdown() {
	alive.store(false);
	// TODO: notify queue waiters
}

void AsyncTaskChecker::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

void AsyncTaskChecker::executorMain(int executorId) {
	// TODO: acquire core, bind to core, set thread priority
	util::initializeCudaStream();

	std::vector<AsyncTask*> pending_tasks;
	while (alive.load()) {
		AsyncTask* next;
		while (queue.try_pop(next)) {
			pending_tasks.push_back(next);
		}

		std::vector<AsyncTask*> still_pending;
		for (AsyncTask* task : pending_tasks) {
			if (task->is_complete()) {
				task->process_completion();
			} else {
				still_pending.push_back(task);
			}
		}
		pending_tasks = still_pending;
	}
}


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