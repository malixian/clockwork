#ifndef _CLOCKWORK_DECOUPLEDRUNTIME_H_
#define _CLOCKWORK_DECOUPLEDRUNTIME_H_

#include <cuda_runtime.h>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <clockwork/basic_threadpool.h>
#include "clockwork/runtime.h"
#include "tbb/concurrent_queue.h"

namespace clockwork {

/**
The decoupled runtime has an executor for each resource type.

An executor consists of a basic_threadpool that executes std::functions in the
order they were enqueued

each parameter specifies the size of the threadpool for the corresponding executor

Threadpools do not block on asynchronous cuda work.
**/
Runtime* newDecoupledRuntime(const int disk_load_threads = 1,
                             const int cpu_threads = 1,
                             const int upload_params_threads = 1,
                             const int input_threads = 1,
                             const int gpu_threads = 1,
                             const int output_threads = 1,
                             const int out_proc_threads = 1
                           );

namespace decoupledruntime {


class Task {
public:
	TaskType type;
	std::function<void(void)> operation;
  std::atomic<bool> asyncStarted;
  std::atomic<bool> nextHasCecked;
	cudaEvent_t asyncComplete;
	Task* prev = nullptr;
	Task* next = nullptr;

	Task(TaskType type, std::function<void(void)> f);
};

class Executor {
private:
  ThreadPool tp_;

public:
	const TaskType type;

	Executor(TaskType type, const unsigned numThreads);

  void addTask(Task* t);
};

class DecoupledRuntime : public clockwork::Runtime {
public:
  DecoupledRuntime(const int disk_load_threads,
                   const int cpu_threads,
                   const int upload_params_threads,
                   const int input_threads,
                   const int gpu_threads,
                   const int output_threads,
                   const int out_proc_threads) : disk_(TaskType::Disk, disk_load_threads),
                                                 cpu_(TaskType::CPU, cpu_threads),
                                                 load_to_device_(TaskType::PCIe_H2D_Weights, upload_params_threads),
                                                 upload_inputs_(TaskType::PCIe_H2D_Inputs, input_threads),
                                                 gpu_(TaskType::GPU, gpu_threads),
                                                 d2h_pcie_(TaskType::PCIe_D2H_Output, output_threads),
                                                 out_proc_(TaskType::Sync, out_proc_threads) {
    // do i need something here? kinda drunk help
  }

  virtual clockwork::RequestBuilder* newRequest();

  virtual void enqueue(std::vector<Task*>& tasks);
  void shutdown(bool awaitShutdown);
	void join();
private:
  // model loaded to cpu in separate pipeline
  std::mutex cpuExecLock_;
  Executor disk_; // loading model files from disk
  Executor cpu_; // preprocessing model

  // Executors for each resource and a lock to ensure correct ordering
  std::mutex execLock_;
  // two executors for h2d_pcie
  Executor load_to_device_; // upload model params, setup functions, etc..
  Executor upload_inputs_; // copy input for inference

  Executor gpu_; // execute on gpu
  Executor d2h_pcie_; // copy outputs to host
  Executor out_proc_; // any post processing needed after output is copied back
};

class RequestBuilder : public clockwork::RequestBuilder {
private:
	DecoupledRuntime* runtime;
	std::vector<Task*> tasks;
public:
	RequestBuilder(DecoupledRuntime *runtime);

	virtual RequestBuilder* addTask(TaskType type, std::function<void(void)> operation);

	virtual void submit();
};


}
}

#endif
