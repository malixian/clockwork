#include "clockwork/api/worker_api.h"
#include "clockwork/runtime.h"

namespace clockwork {

Executor::Executor(TaskType type, int num_threads) : alive(true), type(type){
	for (unsigned i = 0; i < num_threads; i++) {
		threads.push_back(std::thread(&Executor::executorMain, this, i));
	}
}

void Executor::enqueue(Task* task) {
	if (!queue.enqueue(task, task->eligible())) {
		throw TaskError(actionErrorShuttingDown, "Cannot enqueue task to executor that is shutting down");
	}
}

void Executor::shutdown() {
	queue.shutdown();
	alive.store(false);
}

void Executor::join() {
	for (unsigned i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

void Executor::executorMain(int executorId) {
	// TODO: acquire core, bind to core, set thread priority
	util::initializeCudaStream();
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


AsyncTaskChecker::AsyncTaskChecker() : alive(true) {
	unsigned num_threads = 1;
	for (unsigned i = 0; i < num_threads; i++) {
		threads.push_back(std::thread(&AsyncTaskChecker::executorMain, this, i));
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

void AsyncTaskChecker::executorMain(int executorId) {
	// TODO: acquire core, bind to core, set thread priority
	util::initializeCudaStream();

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
	gpu_executor->shutdown();
	outputs_executor->shutdown();
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
	gpu_executor->join();
	outputs_executor->join();

	/*
	Only now do we stop the checker.  Async tasks might still be
	outstanding, and we still want to wait for them to complete
	*/
	checker->shutdown();
	checker->join();
}

}