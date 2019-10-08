#include "clockwork/runtime.h"

namespace clockwork {

Executor::Executor(TaskType type, int num_threads) : alive(true), type(type){
	for (unsigned i = 0; i < num_threads; i++) {
		threads.push_back(std::thread(&Executor::executorMain, this, i));
	}
}

void Executor::enqueue(Task* task) {
	queue.enqueue(task, task->eligible());
}

void Executor::shutdown() {
	alive.store(false);
	queue.shutdown();
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

		next->telemetry->dequeued = util::hrt();
		next->run(stream);
		next->telemetry->exec_complete = util::hrt();
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
	// TODO: notify queue waiters
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
	while (alive.load()) {
		AsyncTask* next;
		while (queue.try_pop(next)) {
			pending_tasks.push_back(next);
		}

		std::vector<AsyncTask*> still_pending;
		for (AsyncTask* task : pending_tasks) {
			if (task->is_complete()) {
				task->process_completion();
			} else {
				still_pending.push_back(task);
			}
		}
		pending_tasks = still_pending;
	}
}

}