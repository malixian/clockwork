#ifndef _CLOCKWORK_QUEUE_H_
#define _CLOCKWORK_QUEUE_H_

#include <cuda_runtime.h>
#include <vector>
#include <atomic>
#include "clockwork/util.h"
#include "tbb/concurrent_queue.h"
#include "tvm/runtime/cuda_common.h"


namespace clockwork {


class Request {
public:
	const Model* model;
	// User, account, etc. can also go here

	Request(Model* model) : model(model) {} 
};


class TaskPrecondition {
public:
	virtual void awaitCompletion() = 0;
}


class Task {
private:
    const std::function<void(void)> operation;


	Task(Request* request, Task* previousTask, Executor* executor, std::function<void(void)> operation) : request(request), previousTask(previousTask), executor(executor), operation(operation) {}

public:
	const Request* request;
	const Executor* executor;

	const Task* previousTask;

	const TaskBarrier* barrier = new TaskBarrier(); // Used to signal completion of the task

	Task(Request* request 

	void execute() {
		// Block until the synchronous portion of the previous task has completed
		if (previousTask != nullptr) {
			previousTask->awaitSyncCompletion();
		}
		// Optional: implement task variant that awaits async completion too

	}

	/* Blocks until the synchronous portion of the previous task has completed */
	void awaitParentSyncWorkCompletion() {

	}

	/* Blocks until both the synchronous and asynchronous work of the previous task has completed. */
	void awaitParentCompletion() {

	}

	/* Indicates that this task's synchronous work has completed, and asynchronous work has been queued. */ 
	void markSyncWorkComplete() {

	}

};


}



// #endif