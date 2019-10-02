#ifndef _CLOCKWORK_RUNTIME_H_
#define _CLOCKWORK_RUNTIME_H_

#include <functional>
#include <array>
#include "clockwork/telemetry.h"

namespace clockwork {
	
enum TaskType {
	PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_Output
};
extern std::array<TaskType, 4> TaskTypes;

// enum TaskType {
// 	Disk, CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_Output, Sync, ModuleLoad
// };
// extern std::array<TaskType, 8> TaskTypes;

std::string TaskTypeName(TaskType type);

class RequestBuilder {
public:
	virtual RequestBuilder* setTelemetry(RequestTelemetry* telemetry) = 0;;
	virtual RequestBuilder* addTask(TaskType type, std::function<void(void)> operation, uint64_t eligible) = 0;
	virtual RequestBuilder* setCompletionCallback(std::function<void(void)> onComplete) = 0;
	virtual void submit() = 0;
};

class Runtime {
public:
	virtual RequestBuilder* newRequest() = 0;
	virtual void shutdown(bool awaitShutdown) = 0;
	virtual void join() = 0;
};

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
Runtime* newRuntime(const unsigned numThreadsPerExecutor, const unsigned maxOutstandingPerExecutor);




}

#endif
