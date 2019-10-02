#include "clockwork/clockworkruntime.h"
#include "tvm/runtime/cuda_common.h"
#include "clockwork/runtime.h"
#include "clockwork/util.h"
#include <dmlc/logging.h>
#include <array>

namespace clockwork {

Runtime* newClockworkRuntime(const unsigned numThreadsPerExecutor, const unsigned maxOutstandingPerExecutor) {
	return new clockworkruntime::ClockworkRuntime(numThreadsPerExecutor, maxOutstandingPerExecutor);
}

namespace clockworkruntime {

Task::Task(TaskType type, std::function<void(void)> f) : type(type), f(f), syncComplete(false), eligible(0) {
	CUDA_CALL(cudaEventCreate(&asyncComplete));
	this->telemetry = new TaskTelemetry();
}

Task::Task(TaskType type, std::function<void(void)> f, uint64_t eligible) : type(type), f(f), syncComplete(false), eligible(eligible) {
	CUDA_CALL(cudaEventCreate(&asyncComplete));
	this->telemetry = new TaskTelemetry();
}

void Task::awaitCompletion() {
	while (!syncComplete.load()); // Busy-wait on sync part
	CUDA_CALL(cudaEventSynchronize(asyncComplete)); // Busy-wait on async part
}

bool Task::isSyncComplete() {
	return syncComplete.load();
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

void Task::run() {
	cudaStream_t stream = tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream;

	if (prev != nullptr) {
		while (!prev->isSyncComplete()); // Busy wait, but only for sync part

		// Async part of this task must wait on async part of previous task
		CUDA_CALL(cudaStreamWaitEvent(stream, prev->asyncComplete, 0));
	}

	f();

	CUDA_CALL(cudaEventRecord(asyncComplete, stream));
	syncComplete.store(true);
}

Executor::Executor(TaskType type, const unsigned numThreads, const unsigned maxOutstanding) : alive(true), type(type), maxOutstanding(maxOutstanding) {
	for (unsigned i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&Executor::executorMain, this, i));
	}
}

void Executor::enqueue(Task* task) {
	queue.enqueue(task, task->eligible);
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
	util::initializeCudaStream();

	std::vector<Task*> pending;
	while (alive.load()) {
		// Finish any pending tasks that are complete
		for (unsigned i = 0; i < pending.size(); i++) {
			if (pending[i]->isAsyncComplete()) {
				if (pending[i]->onComplete != nullptr) {
					pending[i]->onComplete();
				}
				pending.erase(pending.begin()+i);
				i--;
			}
		}

		if (pending.size() == maxOutstanding) {
			// Too many outstanding async tasks
			continue;
		}

		Task* next;
		if (pending.size() == 0) {
			// Just block on the queue
			next = queue.dequeue();
		} else if (!queue.try_dequeue(next)) {
			// Nothing immediately available, don't block because still pending to clear
			continue;
		}

		if (next != nullptr) {
			next->run();
			pending.push_back(next);
		}
	}
}

ClockworkRuntime::ClockworkRuntime(const unsigned numThreads, const unsigned maxOutstanding) : alive(true), numThreads(numThreads), maxOutstanding(maxOutstanding), executors(TaskTypes.size()) {
	for (unsigned i = 0; i < TaskTypes.size(); i++) {
		executors[TaskTypes[i]] = new Executor(TaskTypes[i], numThreads, maxOutstanding);
	}
}

ClockworkRuntime::~ClockworkRuntime() {
	shutdown(false);
}

void ClockworkRuntime::enqueue(Task* task) {
	executors[task->type]->enqueue(task);
}

void ClockworkRuntime::shutdown(bool awaitShutdown) {
	alive.store(false);
	for (unsigned i = 0; i < executors.size(); i++) {
		executors[i]->shutdown();
	}
	if (awaitShutdown) {
		join();
	}
}

void ClockworkRuntime::join() {
	for (unsigned i = 0; i < executors.size(); i++) {
		executors[i]->join();
	}
}

clockwork::RequestBuilder* ClockworkRuntime::newRequest() {
	return new RequestBuilder(this);
}

RequestBuilder::RequestBuilder(ClockworkRuntime *runtime) : runtime(runtime) {}

RequestBuilder* RequestBuilder::setTelemetry(RequestTelemetry* telemetry) {
	this->telemetry = telemetry;
}

RequestBuilder* RequestBuilder::addTask(TaskType type, std::function<void(void)> operation) {
	tasks.push_back(new Task(type, operation));
	return this;
}

RequestBuilder* RequestBuilder::addTask(TaskType type, std::function<void(void)> operation, uint64_t eligible) {
	tasks.push_back(new Task(type, operation, eligible));
	return this;
}

RequestBuilder* RequestBuilder::setCompletionCallback(std::function<void(void)> onComplete) {
	this->onComplete = onComplete;
}

void RequestBuilder::submit() {
	CHECK(telemetry != nullptr) << "RequestBuilder requires a RequestTelemetry to be set using setTelemetry";
	
	// Link the tasks
	for (unsigned i = 1; i < tasks.size(); i++) {
		tasks[i-1]->next = tasks[i];
		tasks[i]->prev = tasks[i-1];
	}

	// Add the telemetry
	for (auto &task : tasks) {
		this->telemetry->tasks.push_back(task->telemetry);
	}

	// Enqueue all tasks
	for (unsigned i = 0; i < tasks.size(); i++) {
		runtime->enqueue(tasks[i]);
	}

	if (tasks.size() > 0) {
		tasks[tasks.size()-1]->onComplete = onComplete;
	}
}

}
}
