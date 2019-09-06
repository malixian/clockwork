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

/**
The Clockwork runtime has an executor for each resource type.

An executor consists of a self-contained threadpool and queue.

numThreadsPerExecutor specifies the size of the threadpool

Threadpools do not block on asynchronous cuda work.  Use maxOutstandingPerExecutor to specify
a maximum number of incomplete asynchronous tasks before an executor will block.

Unlike the Greedy executor, all tasks are enqueued to all executors immediately.
Tasks are assigned a priority, and each executor uses a priority queue.
Each task has an eligibility time, which represents the earliest point they are allowed to execute.
If a task becomes eligible, is dequeued by an executor, but its predecessor task hasn't completed, then the executor blocks.
**/
Runtime* newClockworkRuntime(const unsigned numThreadsPerExecutor, const unsigned maxOutstandingPerExecutor);

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
	ClockworkRuntime* runtime;
	std::vector<Task*> tasks;
public:
	RequestBuilder(ClockworkRuntime *runtime);

	virtual RequestBuilder* addTask(TaskType type, std::function<void(void)> operation);
	RequestBuilder* addTask(TaskType type, std::function<void(void)> operation, uint64_t eligible);

	virtual void submit();
};

}
}

#endif
