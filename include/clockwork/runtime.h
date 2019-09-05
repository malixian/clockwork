#ifndef _CLOCKWORK_RUNTIME_H_
#define _CLOCKWORK_RUNTIME_H_

#include <functional>

namespace clockwork {

enum TaskType { Disk, CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_output, Sync };

class RequestBuilder {
public:
	virtual RequestBuilder* addTask(TaskType type, std::function<void(void)> operation) = 0;
	virtual void submit() = 0;
};

class Runtime {
public:
	virtual RequestBuilder* newRequest() = 0;
	virtual void shutdown(bool awaitShutdown) = 0;
	virtual void join() = 0;
};


}

#endif