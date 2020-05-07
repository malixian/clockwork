#include "clockwork/api/worker_api.h"
#include "clockwork/runtime.h"
#include "clockwork/action.h"

namespace clockwork {

void BaseExecutor::enqueue(Task* task) {
	if (!queue.enqueue(task, task->eligible())) {
		throw TaskError(actionErrorShuttingDown, "Cannot enqueue task to executor that is shutting down");
	}
}

void BaseExecutor::shutdown() {
	queue.shutdown();
	alive.store(false);
}

void BaseExecutor::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

CPUExecutor::CPUExecutor(TaskType type, std::vector<unsigned> cores) : BaseExecutor(type) {
	for (unsigned i = 0; i < cores.size(); i++) {
		threads.push_back(std::thread(&CPUExecutor::executorMain, this, i, cores[i]));
	}
}

void CPUExecutor::executorMain(unsigned executor_id, unsigned core) {
	std::cout << TaskTypeName(type) << "-" << executor_id << " binding to core " << core << std::endl;
	util::set_core(core);
	// util::setCurrentThreadMaxPriority();

	while (alive.load()) {
		// TODO: possibility off too many outstanding asyc tasks

		// TODO: queue should spin-wait rather than blocking
		// TODO: shutdown queue or use try_dequeue

		// Currently, CPUExecutor is only used for LoadModelTask
		LoadModelFromDiskTask* next = dynamic_cast<LoadModelFromDiskTask*>(queue.dequeue());
		
		if (next != nullptr) {
			auto telemetry = next->telemetry;
			telemetry->dequeued = util::hrt();
			next->run();
			telemetry->exec_complete = util::hrt();
		}
	}

	std::vector<Task*> tasks = queue.drain();
	for (Task* task : tasks) {
		task->cancel();
	}
}

GPUExecutorShared::GPUExecutorShared(TaskType type, std::vector<unsigned> cores, unsigned num_gpus):
	BaseExecutor(type), num_gpus(num_gpus) {
	for (unsigned i = 0; i < cores.size(); i++) {
		threads.push_back(std::thread(&GPUExecutorShared::executorMain, this, i, cores[i]));
	}
}

void GPUExecutorShared::executorMain(unsigned executor_id, unsigned core) {
	int priority = 0;
	if (type==TaskType::PCIe_H2D_Inputs || type==TaskType::PCIe_D2H_Output) {
		priority = -1;
	}

	std::cout << TaskTypeName(type) << "-" << executor_id << " binding to core " << core << " with GPU priority " << priority << std::endl;
	// util::set_core(core);
	// util::setCurrentThreadMaxPriority();

	std::vector<cudaStream_t> streams;
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		cudaStream_t stream;
		CUDA_CALL(cudaSetDevice(gpu_id));
		CUDA_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
		streams.push_back(stream);
	}

	int prev_gpu_id = -1;
	while (alive.load()) {
		// TODO: possibility off too many outstanding asyc tasks

		// TODO: queue should spin-wait rather than blocking
		// TODO: shutdown queue or use try_dequeue

		Task* next = queue.dequeue();

		if (next != nullptr) {

			// For tasks of type PCIe_H2D_Weights, we do not want two streams
			// to transfer weights in parallel; therefore, whenever there is a
			// a change in the stream, we synchronize host until the previous
			// stream has been entirely flushed out
			if (type == PCIe_H2D_Weights and prev_gpu_id != -1 and prev_gpu_id != next->gpu_id) {
				CUDA_CALL(cudaSetDevice(next->gpu_id));
				CUDA_CALL(cudaStreamSynchronize(streams[prev_gpu_id]));
				prev_gpu_id = next->gpu_id;
			}

			auto telemetry = next->telemetry;
			telemetry->dequeued = util::hrt();
			next->run(streams[next->gpu_id]);
			telemetry->exec_complete = util::hrt();
		}
	}

	std::vector<Task*> tasks = queue.drain();
	for (Task* task : tasks) {
		task->cancel();
	}
}

GPUExecutorExclusive::GPUExecutorExclusive(TaskType type, std::vector<unsigned> cores, unsigned gpu_id):
	BaseExecutor(type), gpu_id(gpu_id) {
	for (unsigned i = 0; i < cores.size(); i++) {
		threads.push_back(std::thread(&GPUExecutorExclusive::executorMain, this, i, cores[i]));
	}
}

void GPUExecutorExclusive::executorMain(unsigned executor_id, unsigned core) {
	std::cout << TaskTypeName(type) << "-" << executor_id << " binding to core " << core << std::endl;
	util::set_core(core);
	// util::setCurrentThreadMaxPriority();
	util::initializeCudaStream(gpu_id);
	cudaStream_t stream = util::Stream();

	while (alive.load()) {
		// TODO: possibility off too many outstanding asyc tasks

		// TODO: queue should spin-wait rather than blocking
		// TODO: shutdown queue or use try_dequeue

		Task* next = queue.dequeue();

		if (next != nullptr) {
			auto telemetry = next->telemetry;

			telemetry->dequeued = util::hrt();
			next->run(stream);
			telemetry->exec_complete = util::hrt();
		}
	}

	std::vector<Task*> tasks = queue.drain();
	for (Task* task : tasks) {
		task->cancel();
	}
}


AsyncTaskChecker::AsyncTaskChecker(std::vector<unsigned> cores) : alive(true) {
	for (unsigned i = 0; i < cores.size(); i++) {
		threads.push_back(std::thread(&AsyncTaskChecker::executorMain, this, i, cores[i]));
	}
}

void AsyncTaskChecker::enqueue(AsyncTask* task) {
	queue.push(task);
}

void AsyncTaskChecker::shutdown() {
	alive.store(false);
}

void AsyncTaskChecker::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

void AsyncTaskChecker::executorMain(unsigned executor_id, unsigned core) {
	std::cout << "Checker-" << executor_id << " binding to core " << core << std::endl;
	util::set_core(core);
	// util::setCurrentThreadMaxPriority();
	//util::initializeCudaStream(GPU_ID_0); // TODO Is this call necessary?

	std::vector<AsyncTask*> pending_tasks;
	while (alive.load() || pending_tasks.size() > 0) {
		// Check completed tasks
		std::vector<AsyncTask*> still_pending;
		for (AsyncTask* task : pending_tasks) {
			if (task->is_complete()) {
				auto telemetry = task->telemetry;
				task->process_completion();
				telemetry->async_complete = util::hrt();
			} else {
				still_pending.push_back(task);
			}
		}
		pending_tasks = still_pending;

		// Drain any newly queued tasks
		AsyncTask* next;
		while (queue.try_pop(next)) {
			pending_tasks.push_back(next);
		}
	}
}

void ClockworkRuntime::shutdown(bool await_completion) {
	/* 
	Stop executors.  They'll finish current tasks, prevent enqueueing
	new tasks, and cancel tasks that haven't been started yet
	*/
	load_model_executor->shutdown();
	weights_executor->shutdown();
	inputs_executor->shutdown();
	outputs_executor->shutdown();
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		gpu_executors[gpu_id]->shutdown();
	}

	if (await_completion) {
		join();
	}
}

void ClockworkRuntime::join() {
	/*
	Wait for executors to be finished
	*/
	load_model_executor->join();
	weights_executor->join();
	inputs_executor->join();
	outputs_executor->join();
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		gpu_executors[gpu_id]->join();
	}

	/*
	Only now do we stop the checker.  Async tasks might still be
	outstanding, and we still want to wait for them to complete
	*/
	checker->shutdown();
	checker->join();
}

}
