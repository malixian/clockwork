#include "clockwork/alternatives/greedyruntime.h"
#include "tvm/runtime/cuda_common.h"
#include "clockwork/runtime.h"
#include "clockwork/util.h"
#include <array>
#include <sstream>

namespace clockwork {

Runtime* newGreedyRuntime(const unsigned numThreadsPerExecutor, const unsigned maxOutstandingPerExecutor) {
	return new greedyruntime::GreedyRuntime(numThreadsPerExecutor, maxOutstandingPerExecutor);
}

namespace greedyruntime {

Task::Task(TaskType type, std::function<void(void)> f) : type(type), f(f) {
	CUDA_CALL(cudaEventCreate(&asyncSubmit));
	CUDA_CALL(cudaEventCreate(&asyncStart));
	CUDA_CALL(cudaEventCreate(&asyncComplete));
	this->telemetry = new TaskTelemetry();
	this->telemetry->created = clockwork::util::hrt();
}

void Task::awaitCompletion() {
	while (!syncComplete.load()); // Busy-wait on sync part
	CUDA_CALL(cudaEventSynchronize(asyncComplete)); // Busy-wait on async part
}

bool Task::isAsyncComplete() {
	cudaError_t status = cudaEventQuery(asyncComplete);
	if (status == cudaErrorNotReady) {
		return false;
	}
	CHECK(status == cudaSuccess || 
		  status == cudaErrorNotReady ||
		  status == cudaErrorCudartUnloading
		 ) << "CUDA: " << cudaGetErrorString(status);
	return true;
}

void Task::run(cudaStream_t execStream, cudaStream_t telemetryStream) {
	telemetry->dequeued = clockwork::util::hrt();

	CUDA_CALL(cudaEventRecord(asyncSubmit, telemetryStream))
	CUDA_CALL(cudaEventRecord(asyncStart, execStream));

	f();
	CUDA_CALL(cudaEventRecord(asyncComplete, execStream));

	CUDA_CALL(cudaStreamSynchronize(execStream));
	telemetry->exec_complete = clockwork::util::hrt();


	syncComplete.store(true);
}

void Task::processAsyncCompleteTelemetry() {
	telemetry->async_complete = clockwork::util::hrt();
	CUDA_CALL(cudaEventElapsedTime(&telemetry->async_wait, asyncSubmit, asyncStart));
	CUDA_CALL(cudaEventElapsedTime(&telemetry->async_duration, asyncStart, asyncComplete));
}

void Task::complete() {
	if (onComplete != nullptr) {
		onComplete();
	}
}

void deleteTaskAndPredecessors(Task* task) {
	while (task != nullptr) {
		Task* prev = task->prev;
		delete task;
		task = prev;
	}
}

Executor::Executor(GreedyRuntime* runtime, TaskType type, const unsigned numThreads, const unsigned maxOutstanding) : runtime(runtime), type(type), maxOutstanding(maxOutstanding) {
	for (unsigned i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&Executor::executorMain, this, i));
	}
}

void Executor::enqueue(Task* task) {
	queue.push(task);
}

void Executor::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

void Executor::executorMain(int executorId) {
	unsigned core = runtime->assignCore(type, executorId);
	util::set_core(core);
	util::setCurrentThreadMaxPriority();
	util::initializeCudaStream();




	cudaStream_t execStream = clockwork::util::Stream();
	cudaStream_t telemetryStream;
	CUDA_CALL(cudaStreamCreate(&telemetryStream));
	std::vector<Task*> pending;

	std::stringstream ss;
	ss << TaskTypeName(type) << "-" << executorId << " core " << core << " streams " << execStream << " " << telemetryStream << std::endl;
	std::cout << ss.str();

	while (runtime->isAlive()) {
		// Finish any pending tasks that are complete
		for (unsigned i = 0; i < pending.size(); i++) {
			if (pending[i]->isAsyncComplete()) {
				pending[i]->processAsyncCompleteTelemetry();
				if (pending[i]->next != nullptr) {
					runtime->enqueue(pending[i]->next);
				} else {
					pending[i]->complete();
					deleteTaskAndPredecessors(pending[i]);
				}
				pending.erase(pending.begin()+i);
				i--;
			}
		}

		// Execute a request if any are queued
		if (pending.size() < maxOutstanding) {
			Task* next;
			if (queue.try_pop(next)) {
				next->run(execStream, telemetryStream);
				pending.push_back(next);
			}
		}
	}
}

GreedyRuntime::GreedyRuntime(const unsigned numThreads, const unsigned maxOutstanding) : alive(true), numThreads(numThreads), maxOutstanding(maxOutstanding), executors(TaskTypes.size()) {
	for (unsigned i = 0; i < TaskTypes.size(); i++) {
		executors[TaskTypes[i]] = new Executor(this, TaskTypes[i], numThreads, maxOutstanding);
	}
}

GreedyRuntime::~GreedyRuntime() {
	shutdown(false);
}

unsigned GreedyRuntime::assignCore(TaskType type, int executorId) {
	unsigned core = (2 + this->coreCount++) % util::get_num_cores();
	CHECK(core >= 2) << "GreedyRuntime ran out of cores";
	return core;
}

void GreedyRuntime::enqueue(Task* task) {
	task->telemetry->task_type = task->type;
	task->telemetry->enqueued = clockwork::util::hrt();
	task->telemetry->eligible_for_dequeue = task->telemetry->enqueued; // Not used in greedy
	executors[task->type]->enqueue(task);
}

void GreedyRuntime::shutdown(bool awaitShutdown) {
	alive.store(false);
	if (awaitShutdown) {
		join();
	}
}

void GreedyRuntime::join() {
	for (unsigned i = 0; i < executors.size(); i++) {
		executors[i]->join();
	}
}

clockwork::RequestBuilder* GreedyRuntime::newRequest() {
	return new RequestBuilder(this);
}

RequestBuilder::RequestBuilder(GreedyRuntime *runtime) : runtime(runtime) {}

RequestBuilder* RequestBuilder::setTelemetry(RequestTelemetry* telemetry) {
	this->telemetry = telemetry;
}

RequestBuilder* RequestBuilder::addTask(TaskType type, std::function<void(void)> operation) {
	tasks.push_back(new Task(type, operation));
	return this;
}

RequestBuilder* RequestBuilder::setCompletionCallback(std::function<void(void)> onComplete) {
	this->onComplete = onComplete;
}

void RequestBuilder::submit() {
	CHECK(telemetry != nullptr) << "RequestBuilder requires a RequestTelemetry to be set using setTelemetry";

	// Initialize and link the tasks
	for (unsigned i = 1; i < tasks.size(); i++) {
		tasks[i-1]->next = tasks[i];
		tasks[i]->prev = tasks[i-1];
	}

	// Add the telemetry
	for (auto &task : tasks) {
		this->telemetry->tasks.push_back(task->telemetry);
	}

	// Enqueue the first task
	if (tasks.size() > 0) {
		tasks[tasks.size()-1]->onComplete = onComplete;
		runtime->enqueue(tasks[0]);
	}
	delete this;
}

}
}