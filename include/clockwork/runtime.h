#ifndef _CLOCKWORK_MANAGER_H_
#define _CLOCKWORK_MANAGER_H_

#include <functional>

namespace clockwork {

enum TaskType { Disk, CPU, PCIe_H2D_Weights, PCIe_H2D_Inputs, GPU, PCIe_D2H_output, Sync };

class RequestBuilder {
public:
	virtual RequestBuilder* addTask(TaskType type, std::function<void(void)> operation) = 0;
	virtual void submit() = 0;
};

class MultitenantRuntime {
public:
	virtual RequestBuilder* newRequest() = 0;
};


}

// #endif