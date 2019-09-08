#include "clockwork/decoupledruntime.h"
#include "tvm/runtime/cuda_common.h"
#include "clockwork/runtime.h"
#include <array>

namespace clockwork {

Runtime* newDecoupledRuntime(const int disk_load_threads,
                             const int cpu_threads,
                             const int upload_params_threads,
                             const int input_threads,
                             const int gpu_threads,
                             const int output_threads,
                             const int out_proc_threads
                           ) {
	return new decoupledruntime::DecoupledRuntime(
		disk_load_threads,
		cpu_threads,
		upload_params_threads,
		input_threads,
		gpu_threads,
		output_threads,
		out_proc_threads
	);
}

namespace decoupledruntime {

// decoupled runtime impl

void DecoupledRuntime::enqueue(std::vector<Task*>& tasks) {
	if (tasks.size() == 2 && tasks[0]->type == TaskType::Disk) { // loadModel
		std::lock_guard<std::mutex> lk(cpuExecLock_);
		disk_.addTask(tasks[0]);
		cpu_.addTask(tasks[1]);
		return;
	} else if (tasks.size() == 4 && tasks[0]->type == TaskType::PCIe_H2D_Inputs) { // infer_
		std::lock_guard<std::mutex> lk(execLock_);
		upload_inputs_.addTask(tasks[0]);
		gpu_.addTask(tasks[1]);
		d2h_pcie_.addTask(tasks[2]);
		out_proc_.addTask(tasks[3]);
		return;
	} else if (tasks.size() == 5 && tasks[0]->type == TaskType::PCIe_H2D_Weights) { // uploadToGpuAndInfer
		std::lock_guard<std::mutex> lk(execLock_);
		load_to_device_.addTask(tasks[0]);
		upload_inputs_.addTask(tasks[1]);
		gpu_.addTask(tasks[2]);
		d2h_pcie_.addTask(tasks[3]);
		out_proc_.addTask(tasks[4]);
		return;
	}

	CHECK(false) << "Invalid operation enqueued into DecoupledRuntime";
}

void DecoupledRuntime::shutdown(bool awaitShutdown) {
	// for right now we default to letting the executors be cleaned up on destruction
}

void DecoupledRuntime::join() {
	// for right now we default to letting the executors be cleaned up on destruction
}

// Task and Executor impl

Task::Task(TaskType type, std::function<void(void)> f) : type(type), operation(f), asyncStarted(false), nextHasCecked(false)  {
	CUDA_CALL(cudaEventCreateWithFlags(&asyncComplete, cudaEventBlockingSync | cudaEventDisableTiming));
}

Executor::Executor(TaskType type, const unsigned numThreads) : type(type), tp_(numThreads) {}

void Executor::addTask(Task* task) {
	std::function<void(void)> exec = [=](){
		if (task->prev != nullptr) {
			// wait for previous task to start
			while (!task->prev->asyncStarted.load()) {}

			// make this stream wait on previous executor stream contents
			CUDA_CALL(cudaStreamWaitEvent(tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream, task->prev->asyncComplete, 0));

			// mark the dependency as done so that other executor can move to the next task
			task->prev->nextHasCecked.store(true);
		}

		task->operation();

		// record content of stream to sync on in the next executor's stream
		CUDA_CALL(cudaEventRecord(task->asyncComplete, tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream));

		// notify the next task that it can at least queue the next operation
		task->asyncStarted.store(true);

		if (task->next != nullptr) {
			while (!task->nextHasCecked.load()) {
			}
		}

		CUDA_CALL(cudaStreamSynchronize(tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream));

		delete task;
	};

	tp_.push(exec);
}


// request creation and submission

clockwork::RequestBuilder* DecoupledRuntime::newRequest() {
	return new RequestBuilder(this);
}

RequestBuilder::RequestBuilder(DecoupledRuntime *runtime) : runtime(runtime) {}

RequestBuilder* RequestBuilder::addTask(TaskType type, std::function<void(void)> operation) {
	tasks.push_back(new Task(type, operation));
	return this;
}

void RequestBuilder::submit() {
	// Initialize and link the tasks
	for (unsigned i = 0; i < tasks.size(); i++) {
		if (i > 0) {
			tasks[i-1]->next = tasks[i];
			tasks[i]->prev = tasks[i-1];
		}
	}

	runtime->enqueue(tasks);
}

}
}
