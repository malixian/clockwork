#ifndef _CLOCKWORK_CLOCKWORKRUNTIME_H_
#define _CLOCKWORK_CLOCKWORKRUNTIME_H_

#include <cuda_runtime.h>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "clockwork/runtime.h"
#include "tbb/concurrent_queue.h"

namespace clockwork {

namespace clockworkruntime {


class Task {
public:
	TaskType type;
	std::function<void(void)> f;
	std::atomic_bool syncComplete;
	cudaEvent_t asyncComplete;
	uint64_t eligible;
	Task* prev = nullptr;
	Task* next = nullptr;
	TaskTelemetry* telemetry;
	std::function<void(void)> onComplete;

	Task(TaskType type, std::function<void(void)> f);
	Task(TaskType type, std::function<void(void)> f, uint64_t eligible);

	void awaitCompletion();
	bool isSyncComplete();
	bool isAsyncComplete();
	void run();
};

class TaskPriorityQueue {
private:

	struct TaskContainer {
		Task* task;

		friend bool operator < (const TaskContainer& lhs, const TaskContainer &rhs) {
			return lhs.task->eligible < rhs.task->eligible;
		}
		friend bool operator > (const TaskContainer& lhs, const TaskContainer &rhs) {
			return lhs.task->eligible > rhs.task->eligible;
		}
	};

	bool alive = true;
	std::mutex mutex;
	std::condition_variable condition;
	std::priority_queue<TaskContainer, std::vector<TaskContainer>, std::greater<TaskContainer>> queue;

public:

	void enqueue(Task* task);
	bool try_dequeue(Task* &task);
	Task* dequeue();
	void shutdown();
};

class Executor {
private:
	std::atomic_bool alive;
	TaskPriorityQueue queue;
	std::vector<std::thread> threads;


public:
	const TaskType type;
	const unsigned maxOutstanding;

	Executor(TaskType type, const unsigned numThreads, const unsigned maxOutstanding);

	void enqueue(Task* task);
	void shutdown();
	void join();
	void executorMain(int executorId);
};

class ClockworkRuntime : public clockwork::Runtime {
private:
	std::atomic_bool alive;
	std::vector<Executor*> executors;
	const unsigned numThreads;
	const unsigned maxOutstanding;

public:
	ClockworkRuntime(const unsigned numThreads, const unsigned maxOutstanding);
	~ClockworkRuntime();

	void enqueue(Task* task);
	void shutdown(bool awaitShutdown);
	void join();
	bool isAlive() { return alive.load(); }
	virtual clockwork::RequestBuilder* newRequest();
};

class RequestBuilder : public clockwork::RequestBuilder {
private:
	RequestTelemetry* telemetry = nullptr;
	std::function<void(void)> onComplete = nullptr;
	ClockworkRuntime* runtime;
	std::vector<Task*> tasks;
public:
	RequestBuilder(ClockworkRuntime *runtime);

	RequestBuilder* setTelemetry(RequestTelemetry* telemetry);
	RequestBuilder* addTask(TaskType type, std::function<void(void)> operation);
	RequestBuilder* addTask(TaskType type, std::function<void(void)> operation, uint64_t eligible);
	RequestBuilder* setCompletionCallback(std::function<void(void)> onComplete);

	void submit();
};

}
}

#endif
