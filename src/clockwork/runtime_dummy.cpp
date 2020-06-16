#include "clockwork/api/worker_api.h"
#include "clockwork/runtime_dummy.h"
#include "clockwork/action_dummy.h"
#include "clockwork/thread.h"

namespace clockwork {

void BaseExecutor_noTelemetry::enqueue(Task* task) {
	if (!queue.enqueue(task, task->eligible())) {
		throw TaskError(actionErrorShuttingDown, "Cannot enqueue task to executor that is shutting down");
	}
}

void BaseExecutor_noTelemetry::shutdown() {
	queue.shutdown();
	alive.store(false);
}

void BaseExecutor_noTelemetry::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

CPUExecutor_noTelemetry::CPUExecutor_noTelemetry(TaskType type) : BaseExecutor_noTelemetry(type) {
	threads.push_back(std::thread(&CPUExecutor_noTelemetry::executorMain, this, 0));
	for (auto &thread : threads) threading::initGPUThread(0, thread);
}

void CPUExecutor_noTelemetry::executorMain(unsigned executor_id) {
	std::cout << TaskTypeName(type) << "-" << executor_id << " started" << std::endl;

	while (alive.load()) {
		// Currently, CPUExecutor is only used for LoadModelTask
		LoadModelFromDiskTask* next = dynamic_cast<LoadModelFromDiskTask*>(queue.dequeue());
		
		if (next != nullptr) {
			//auto telemetry = next->telemetry;
			//telemetry->dequeued = util::hrt();
			next->run();
			//telemetry->exec_complete = util::hrt();
		}
	}

	std::vector<Task*> tasks = queue.drain();
	for (Task* task : tasks) {
		task->cancel();
	}
}

SingleThreadExecutor::SingleThreadExecutor(TaskType type, unsigned gpu_id):
	BaseExecutor_noTelemetry(type), gpu_id(gpu_id) {
	threads.push_back(std::thread(&SingleThreadExecutor::executorMain, this, 0));
	for (auto &thread : threads) threading::initGPUThread(gpu_id, thread);
}

void SingleThreadExecutor::enqueueRun(AsyncTask* task) {
	runqueue.push(task);
}

void SingleThreadExecutor::executorMain(unsigned executor_id) {
	std::cout << "GPU" << gpu_id << "-" << TaskTypeName(type) << "-" << executor_id << " started" << std::endl;

//TODO WEI 
	int priority = 0;
	if (type==TaskType::PCIe_H2D_Inputs || type==TaskType::PCIe_D2H_Output) {
		priority = -1;
	}
	util::initializeCudaStream(gpu_id, priority);

	cudaStream_t stream = util::Stream();
//TODO WEI 

	std::vector<AsyncTask*> pending_tasks;
	while (alive.load() || pending_tasks.size() > 0) {

	// Process all pending results
		// Check completed tasks
		std::vector<AsyncTask*> still_pending;
		for (AsyncTask* task : pending_tasks) {
			if (task->is_complete()) {
				//auto telemetry = task->telemetry;
				//telemetry->async_complete = util::hrt();
				task->process_completion();
			} else {
				still_pending.push_back(task);
			}
		}
		pending_tasks = still_pending;

		// Drain any newly queued tasks
		AsyncTask* nextRun;
		while (runqueue.try_pop(nextRun)) {
			pending_tasks.push_back(nextRun);
		}

	// Run one next request if available
		Task* next = queue.dequeue();

		if (next != nullptr) {
			//auto telemetry = next->telemetry;

			//telemetry->dequeued = util::hrt();
			next->run(stream); //TODO WEI 
			//telemetry->exec_complete = util::hrt();
		}
	}

	std::vector<Task*> tasks = queue.drain();
	for (Task* task : tasks) {
		task->cancel();
	}
}

void ClockworkRuntimeDummy::shutdown(bool await_completion) {
	/* 
	Stop executors.  They'll finish current tasks, prevent enqueueing
	new tasks, and cancel tasks that haven't been started yet
	*/
	load_model_executor->shutdown();
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		executors[gpu_id]->shutdown();
	}

	if (await_completion) {
		join();
	}
}

void ClockworkRuntimeDummy::join() {
	/*
	Wait for executors to be finished
	*/
	load_model_executor->join();
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		executors[gpu_id]->join();
	}

}

}
