#ifndef _CLOCKWORK_GREEDYRUNTIME_H_
#define _CLOCKWORK_GREEDYRUNTIME_H_

#include <cuda_runtime.h>
#include <functional>
#include <thread>
#include <atomic>
#include "clockwork/runtime.h"
#include "tbb/concurrent_queue.h"

namespace clockwork {
	
namespace greedyruntime {


class Task {
public:
	TaskType type;
	std::function<void(void)> f;
	std::atomic_bool syncComplete;
	cudaEvent_t asyncComplete;
	Task* prev = nullptr;
	Task* next = nullptr;

	Task(TaskType type, std::function<void(void)> f);

	void awaitCompletion();
	bool isAsyncComplete();
	void run();
};

class GreedyRuntime;

class Executor {
private:
	GreedyRuntime* runtime;
	tbb::concurrent_queue<Task*> queue;
	std::vector<std::thread> threads;


public:
	const TaskType type;
	const unsigned maxOutstanding;

	Executor(GreedyRuntime* runtime, TaskType type, const unsigned numThreads, const unsigned maxOutstanding);

	void enqueue(Task* task);
	void join();
	void executorMain(int executorId);
};

class GreedyRuntime : public clockwork::Runtime {
private:
	std::atomic_bool alive;
	std::vector<Executor*> executors;
	const unsigned numThreads;
	const unsigned maxOutstanding;

public:
	GreedyRuntime(const unsigned numThreads, const unsigned maxOutstanding);
	~GreedyRuntime();

	void enqueue(Task* task);
	void shutdown(bool awaitShutdown);
	void join();
	bool isAlive() { return alive.load(); }
	virtual clockwork::RequestBuilder* newRequest();
};

class RequestBuilder : public clockwork::RequestBuilder {
private:
	GreedyRuntime* runtime;
	std::vector<Task*> tasks;
public:
	RequestBuilder(GreedyRuntime *runtime);

	virtual RequestBuilder* addTask(TaskType type, std::function<void(void)> operation);

	virtual void submit();
};

}
}

#endif