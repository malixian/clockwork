// #ifndef _CLOCKWORK_QUEUE_H_
// #define _CLOCKWORK_QUEUE_H_

// #include <cuda_runtime.h>
// #include <vector>
// #include <atomic>
// #include "clockwork/util.h"
// #include "tbb/concurrent_queue.h"
// #include "tvm/runtime/cuda_common.h"
// #include <type_traits>
// #include <mutex>
// #include <condition_variable>
// #include <queue>


// namespace clockwork {


// /** Used to track completion of a task */
// class Queueable {
// public:
// 	virtual bool isComplete() = 0;
// };

// template<typename T, typename std::enable_if<std::is_base_of<Queueable, T>::value>::type* = nullptr> class Queue {
// public:
// 	virtual T dequeue() = 0;
// };

// template<typename T> class FIFOQueue : public Queue<T> {
// private:
// 	tbb::concurrent_queue<T> queue;

// public:

// 	void enqueue(const T element) {
// 		queue.push(element);
// 	}

// 	virtual T dequeue() {
// 		T task;
// 		while (!queue.try_pop(task));
// 		return task;
// 	}

// };

// /** A FIFO queue with a max number of oustanding tasks */
// template<typename T, int N> class MaxFIFOQueue : public Queue<T> {
// private:
// 	tbb::concurrent_queue<T> queue;

// 	std::vector<T> outstanding;
// 	std::mutex outstanding_mutex;


// public:

// 	void enqueue(const T element) {
// 		queue.push(element);
// 	}

// 	virtual T dequeue() {
// 		std::lock_guard<std::mutex> lock(outstanding_mutex);

// 		// Wait until an outstanding task is done
// 		while (outstanding.size() == N) {
// 			for (int i = 0; i < outstanding.size(); i++) {
// 				if (outstanding[i].isComplete()) {
// 					outstanding.erase(outstanding.begin()+i);
// 					break;
// 				}
// 			}
// 		}

// 		// Now wait for a new task
// 		T task;
// 		while (!queue.try_pop(task));
// 		outstanding.push_back(task);
// 		return task;
// 	}

// };


// /** A Priority queue with a max number of oustanding tasks */
// template<typename T, int N> class MaxPriorityQueue : public Queue<T> {
// private:
// 	class PriorityContainer {
// 	public:
// 		uint64_t priority;
// 		T element;
// 		PriorityContainer(T element, uint64_t priority): element(element), priority(priority) {}

// 		friend bool operator < (const PriorityContainer& lhs, const PriorityContainer &rhs) {
// 			return lhs.priority < rhs.priority;
// 		}
// 		friend bool operator > (const PriorityContainer& lhs, const PriorityContainer &rhs) {
// 			return lhs.priority > rhs.priority;
// 		}
// 	};

// 	std::mutex queue_mutex;
// 	std::condition_variable queue_condition;
// 	std::priority_queue<PriorityContainer, std::vector<PriorityContainer>, std::greater<PriorityContainer>> queue;
// 	//std::priority_queue<PriorityContainer> queue;

// 	std::vector<T> outstanding;
// 	std::mutex outstanding_mutex;


// public:

// 	void enqueue(const T element, uint64_t priority) {
// 		std::unique_lock<std::mutex> lock(queue_mutex);
// 		queue.push(PriorityContainer(element, priority));
// 		queue_condition.notify_all();
// 	}

// 	virtual T dequeue() {
// 		std::lock_guard<std::mutex> l1(outstanding_mutex);

// 		// Wait until an outstanding task is done
// 		while (outstanding.size() == N) {
// 			for (int i = 0; i < outstanding.size(); i++) {
// 				if (outstanding[i].isComplete()) {
// 					outstanding.erase(outstanding.begin()+i);
// 					break;
// 				}
// 			}
// 		}

// 		// Now wait for a new task
// 		std::unique_lock<std::mutex> l2(queue_mutex);
// 		while (queue.empty()) {
// 			queue_condition.wait(l2);
// 		}
// 		T element = queue.top().element;
// 		queue.pop();
// 		outstanding.push_back(element);
// 		return element;
// 	}

// };







// // class SynchronousTaskCompletionTracker : public TaskCompletionTracker {
// // private:
// // 	std::atomic<bool> sync;
// // public:
// // 	virtual bool isComplete() {
// // 		return sync.load();
// // 	}
// // }

// // class AsynchronousTaskCompletionTracker : public TaskCompletionTracker {

// // }



// // /** Typical clockwork tasks decompose into two pieces: synchronous CPU work, 
// // followed by asynchronous GPU work.  Once the synchronous CPU work has 
// // completeed, and the asynchronous GPU work has been scheduled, the caller is
// // expected to call methods on the task barrier to indicate the current status.
// // There are methods so that others can later check to see whether the asynchronous
// // GPU work has completed */
// // class TaskBarrier {
// // private:
// // 	std::atomic<bool> sync; // has the synchronous work completed
// // 	cudaEvent_t async;      // signalled when the asynchronous work is completed

// // public:
// // 	TaskBarrier() {
// // 		CUDA_CALL(cudaEventCreate(&async));
// // 		/** Could create event using cudaEventBlockingSync flag, but we'd prefer
// // 		to busy-wait on events */
// // 	}

// // 	/** Indicates that synchronous CPU work has completed, and asynchronous
// // 	GPU work has been enqueued */
// // 	void markSyncCompletion() {
// // 		CUDA_CALL(cudaEventRecord(async, ManagedCUDAThreadEntry::ThreadLocal()->stream));
// // 		sync.store(true);
// // 	}

// // 	bool isSyncComplete() {
// // 		return sync.load();
// // 	}

// // 	bool isAsyncComplete() {
// // 		cudaError_t status = cudaEventQuery(async);
// // 		if (status == cudaErrorNotReady) {
// // 			return false;
// // 		}
// // 		CHECK(status == cudaSuccess || 
// // 			  status == cudaErrorNotReady ||
// // 			  status == cudaErrorCudartUnloading
// // 			 ) << "CUDA: " << cudaGetErrorString(status);
// // 	}

// // 	void awaitCompletion() {
// // 		while (!sync.load()); // Busy-wait on sync
// // 		CUDA_CALL(cudaEventSynchronize(async)); // Busy-wait on cuda event
// // 	}
// // }



// // /** 
// // template <typename T> class TaskContainer {

// // }

// // /** Interface for clockwork task queues.  The interface only has a dequeue method,
// // because the implementations vary in terms of the information provided to the 
// // enqueue method. */
// // template <typename T> class Queue {
// // public:
// // 	/** Dequeue the next task.  If there are no tasks queued, blocks until one
// // 	is available.  Also creates a completionEvent, that should be used to signal
// // 	task completion. */
// // 	virtual void dequeue(T &task, cudaEvent_t &completionEvent) = 0;

// // };

// // /** A simple thread-safe FIFO queue that uses atomics for mutual exclusion */
// // template <typename T> class FIFOQueue : public Queue<T> {
// // private:
// // 	tbb::concurrent_queue<T> queue;

// // public:
// // 	void enqueue(const T element) {
// // 		queue.push(element);
// // 	}

// // 	void dequeue(T &task, cudaEvent_t &completionEvent) {
// // 		while (!queue.try_pop(task));
// // 		CUDA_CALL(cudaEventCreate(&completionEvent));
// // 	}
// // };

// // /** Represents a fixed-size set of cuda events.  Events are manually added to the set,
// // and lazily removed from the set once the cuda event has been fired. */
// // template <unsigned N> class CudaEventSet {
// // private:
// // 	std::atomic<cudaEvent_t*> events[N];

// // public:
// // 	/**  Creates and returns a new cudaEvent_t, adding it to the set.  If the set
// // 	is at capacity, this method blocks until space is available. */
// // 	cudaEvent_t newEvent() {
// // 		cudaEvent_t* event = new cudaEvent_t();
// // 		CUDA_CALL(cudaEventCreateWithFlags(event, cudaEventBlockingSync));


// // 		cudaError_t sss = cudaEventQuery(*event);
// // 		if (sss == cudaSuccess) {
// // 			std::cout << "hmmmm" << std::endl;
// // 		}

// // 		while (true) {
// // 			// Busy-loop searching for an empty slot
// // 			for (unsigned i = 0; i < N; i++) {
// // 				// Remove existing event from this slot
// // 				cudaEvent_t* slot = events[i].exchange(event);

// // 				if (slot == nullptr) {
// // 					std::cout << "slot empty" << std::endl;
// // 					return *event;
// // 				}

// // 				// Query event status
// // 				cudaError_t status = cudaEventQuery(*slot);
// // 				if (status == cudaErrorNotReady) {
// // 					std::cout << "slot cudaErrorNotReady" << std::endl;
// // 					// Event isn't complete, put it back
// // 					while (!events[i].compare_exchange_weak(event, slot));
// // 				} else {
// // 					CHECK(status == cudaSuccess) << "CUDA: " << cudaGetErrorString(status);
// // 					std::cout << "slot done" << std::endl;

// // 					// Clean up old event
// // 					CUDA_CALL(cudaEventDestroy(*slot));
// // 					delete slot;

// // 					return *event;
// // 				}
// // 			}
// // 		}
// // 	}

// // };


// // /** A simple thread-safe FIFO queue that uses atomics for mutual exclusion.
// // The queue also only allows a fixed number N of outstanding requests.
// // Request completion must be signalled by the caller using the provided CUDA event */
// // template <typename T, unsigned N> class MaxConcurrencyFIFOQueue : public Queue<T> {
// // private:
// // 	tbb::concurrent_queue<T> queue;
// // 	CudaEventSet<N> outstanding_tasks;

// // public:

// // 	void enqueue(const T element) {
// // 		queue.push(element);
// // 	}

// // 	void dequeue(T &task, cudaEvent_t &completionEvent) {
// // 		// Blocks until space is available BEFORE dequeueing task
// // 		completionEvent = outstanding_tasks.newEvent();
// // 		while (!queue.try_pop(task));
// // 	}
// // };



// // template <typename T> class PriorityQueue : public Queue<T> {

// // public:

// // 	/** Enqueue using the current clock as priority */
// // 	void enqueue(const T element) {
// // 		enqueue(element, util::now());
// // 	}

// // 	void enqueue(const T element, int priority) {

// // 	}

// // 	void dequeue(T &task, cudaEvent_t &completionEvent) {

// // 	}

// // 	static void saysomething() {
// // 		std::cout << "hello world: " << util::now() << std::endl;
// // 	}

// // };


// }



// #endif