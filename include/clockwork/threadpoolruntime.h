#ifndef _CLOCKWORK_THREADPOOLMANAGER_H_
#define _CLOCKWORK_THREADPOOLMANAGER_H_

#include <functional>
#include "manager.h"
#include "tbb/concurrent_queue.h"
#include <thread>

namespace clockwork {
namespace threadpoolruntime {


struct ThreadPoolTask {
	TaskType type;
	std::function<void(void)> f;
}

struct ThreadPoolRequest {
	std::vector<ThreadPoolTask> tasks;
}

class ThreadPoolManager : public Manager; // Forward declaration

class ThreadPoolRequestBuilder : public RequestBuilder {
private:
	ThreadPoolManager* manager;
	ThreadPoolRequest* request;

public:

	ThreadPoolRequestBuilder(ThreadPoolManager *manager) : manager(manager), request(new ThreadPoolRequest()) {}

	virtual RequestBuilder* addTask(TaskType type, std::function<void(void)> operation) {
		request->tasks.push_back(ThreadPoolTask{type, operation});
		return this;
	}

	virtual void submit() {
		manager->submit(request);
	}

}

class ThreadPoolQueue {
public:
	virtual void enqueue(ThreadPoolRequest* request) = 0;
	virtual bool try_dequeue(ThreadPoolRequest* &request) = 0;
}

class FIFOThreadPoolQueue : public ThreadPoolQueue {
private:
	tbb::concurrent_queue<ThreadPoolRequest*> queue;

public:

	virtual void enqueue(ThreadPoolRequest* request) {
		queue.push(element);
	}

	virtual bool try_dequeue(ThreadPoolRequest* &request) {
		return queue.try_pop(request);
	}
}

enum ThreadPoolQueueType { FIFO };


/** A simple manager based on a threadpool.
Each threadpool thread executes the request in its entirety */
class MultitenantThreadpoolRuntime : public MultitenantRuntime {
private:
	const int numThreads;
	const ThreadPoolQueue* queue;
	std::atomic_bool alive;
	std::vector<std::thread> threads;

	MultitenantThreadpoolRuntime(const unsigned numThreads, const ThreadPoolQueue* queue) : numThreads(numThreads), queue(queue), alive(true) {
		for (unsigned i = 0; i < numThreads; i++) {
			threads.push_back(std::thread(&ThreadPoolManager::threadpoolMain, this, i));
		}
	}

	~MultitenantThreadpoolRuntime() {
		this->shutdown(false);
	}

	void threadpoolMain(int threadNumber) {
		ThreadPoolRequest* request;
		while (alive.load()) { // TODO: graceful shutdown
			if (queue->try_dequeue(request)) {
				for (unsigned i = 0; i < request.tasks.size(); i++) {
					request.tasks[i]();
				}
				delete request;
			}
		}
	}

	void shutdown(bool awaitShutdown) {
		alive.store(false);
		if (awaitShutdown) {
			for (unsigned i = 0; i < threads.size(); i++) {
				threads[i].join();
			}
		}
	}


public:

	static ThreadPoolManager* create(const unsigned numThreads, const ThreadPoolQueueType queueType) {
		ThreadPoolQueue* queue;
		switch (queueType) {
			case FIFO: queue = new FIFOThreadPoolQueue(); break;
		}
		return new ThreadPoolManager(numThreads, queue);
	}

	virtual RequestBuilder* newRequest() {
		return new ThreadPoolRequestBuilder(this);
	}

	void submit(ThreadPoolRequest* request) {
		queue->enqueue(request);
	}

};

}
}

// #endif