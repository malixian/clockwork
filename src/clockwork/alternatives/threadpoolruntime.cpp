
#include "clockwork/alternatives/threadpoolruntime.h"
#include "clockwork/util.h"

namespace clockwork {

Runtime* newFIFOThreadpoolRuntime(const unsigned numThreads) {
	return new threadpoolruntime::ThreadpoolRuntime(numThreads, new threadpoolruntime::FIFOQueue());
}

namespace threadpoolruntime {

RequestBuilder* RequestBuilder::addTask(TaskType type, std::function<void(void)> operation, TaskTelemetry &telemetry) {
	tasks.push_back(Task{type, operation, telemetry});
	return this;
}

void RequestBuilder::submit() {
	submit(nullptr);
}

void RequestBuilder::submit(std::function<void(void)> onComplete) {
	runtime->submit(new Request{tasks, onComplete});
}


void FIFOQueue::enqueue(Request* request) {
	queue.push(request);
}

bool FIFOQueue::try_dequeue(Request* &request) {
	return queue.try_pop(request);
}



ThreadpoolRuntime::ThreadpoolRuntime(const unsigned numThreads, Queue* queue) : numThreads(numThreads), queue(queue), alive(true) {
	for (unsigned i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&ThreadpoolRuntime::threadpoolMain, this, i));
	}
}

ThreadpoolRuntime::~ThreadpoolRuntime() {
	this->shutdown(false);
}

void ThreadpoolRuntime::threadpoolMain(int threadNumber) {
	util::initializeCudaStream();
	Request* request;
	while (alive.load()) { // TODO: graceful shutdown
		if (queue->try_dequeue(request)) {
			for (unsigned i = 0; i < request->tasks.size(); i++) {
				request->tasks[i].f();
			}
			if (request->onComplete != nullptr) {
				request->onComplete();
			}
			delete request;
		}
	}
}

void ThreadpoolRuntime::shutdown(bool awaitShutdown) {
	alive.store(false);
	if (awaitShutdown) {
		join();
	}
}

void ThreadpoolRuntime::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

clockwork::RequestBuilder* ThreadpoolRuntime::newRequest() {
	return new RequestBuilder(this);
}

void ThreadpoolRuntime::submit(Request* request) {
	queue->enqueue(request);
}

}
}