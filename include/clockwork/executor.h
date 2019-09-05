#ifndef _CLOCKWORK_QUEUE_H_
#define _CLOCKWORK_QUEUE_H_

#include <cuda_runtime.h>
#include <vector>
#include <atomic>
#include "clockwork/util.h"
#include "tbb/concurrent_queue.h"
#include "tvm/runtime/cuda_common.h"
#include <type_traits>
#include <mutex>
#include <condition_variable>
#include <queue>


namespace clockwork {

enum TaskType {
	Disk, CPU, PCIe_weights, PCIe_input, GPU, PCIe_output
};


class Request {
public:
	virtual Request addTask(TaskType type, std::function<void(void)> f) = 0;
}


class Executor {
public:

	virtual 

};


}



#endif