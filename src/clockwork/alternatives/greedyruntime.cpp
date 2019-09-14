#include "clockwork/alternatives/greedyruntime.h"
#include "tvm/runtime/cuda_common.h"
#include "clockwork/runtime.h"
#include "clockwork/util.h"
#include <array>

namespace clockwork {

Runtime* newGreedyRuntime(const unsigned numThreadsPerExecutor, const unsigned maxOutstandingPerExecutor) {
	return new greedyruntime::GreedyRuntime(numThreadsPerExecutor, maxOutstandingPerExecutor);
}

namespace greedyruntime {

Task::Task(TaskType type, std::function<void(void)> f) : type(type), f(f) {
	CUDA_CALL(cudaEventCreate(&asyncComplete));
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

void Task::run() {
	f();

	CUDA_CALL(cudaEventRecord(asyncComplete, clockwork::util::Stream()));
	syncComplete.store(true);
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
	util::initializeCudaStream();
	std::vector<Task*> pending;
	while (runtime->isAlive()) {
		// Finish any pending tasks that are complete
		for (unsigned i = 0; i < pending.size(); i++) {
			if (pending[i]->isAsyncComplete()) {
				if (pending[i]->next != nullptr) {
					runtime->enqueue(pending[i]->next);
				} // TODO: final task cleanup / notification
				pending.erase(pending.begin()+i);
				i--;
			}
		}

		// Execute a request if any are queued
		if (pending.size() < maxOutstanding) {
			Task* next;
			if (queue.try_pop(next)) {
				next->run();
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

void GreedyRuntime::enqueue(Task* task) {
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

RequestBuilder* RequestBuilder::addTask(TaskType type, std::function<void(void)> operation) {
	tasks.push_back(new Task(type, operation));
	return this;
}

void RequestBuilder::submit() {
	// Initialize and link the tasks
	for (unsigned i = 0; i < tasks.size(); i++) {
		if (i > 0) {
			tasks[i-1]->next = tasks[i];
			tasks[i]->prev = tasks[i-1];
		}
	}

	// Enqueue the first task
	if (tasks.size() > 0) {
		runtime->enqueue(tasks[0]);
	} // TODO: notify if final task
}

}
}