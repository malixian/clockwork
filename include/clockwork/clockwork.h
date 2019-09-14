#ifndef CLOCKWORK_CLOCKWORK_H_
#define CLOCKWORK_CLOCKWORK_H_

namespace clockwork {

	class Runtime;


	/**
	The threadpool runtime has a fixed-size threadpool for executing requests.
	Each thread executes the entirety of a request at a time, e.g. all the tasks
	of the request.
	**/
	Runtime* newFIFOThreadpoolRuntime(const unsigned numThreads);

	/**
	The Greedy runtime has an executor for each resource type.

	An executor consists of a self-contained threadpool and queue.

	numThreadsPerExecutor specifies the size of the threadpool

	Threadpools do not block on asynchronous cuda work.  Use maxOutstandingPerExecutor to specify
	a maximum number of incomplete asynchronous tasks before an executor will block.
	**/
	Runtime* newGreedyRuntime(const unsigned numThreadsPerExecutor, const unsigned maxOutstandingPerExecutor);

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
	Runtime* newClockworkRuntime(const unsigned numThreadsPerExecutor, const unsigned maxOutstandingPerExecutor);

	  
	static int kEvictionRate = 0;

}

#endif