#ifndef _CLOCKWORK_PRIORITY_QUEUE_H_
#define _CLOCKWORK_PRIORITY_QUEUE_H_

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace clockwork {
	/* This is a priority queue, but one where priorities also define a minimum
time that an enqueued task is eligible to be dequeued.  The queue will block
if no eligible tasks are available */
template <typename T> class time_release_priority_queue {
private:
	struct container {
		T* element;

		// TODO: priority should be a chrono timepoint not the uint64_t, to avoid
		//       expensive conversions.  Or, a different clock altogether
		uint64_t priority;

		friend bool operator < (const container& lhs, const container &rhs) {
			return lhs.priority < rhs.priority;
		}
		friend bool operator > (const container& lhs, const container &rhs) {
			return lhs.priority > rhs.priority;
		}
	};

	std::atomic_bool alive;

	// Unfortunately for now we use mutexes
	std::mutex mutex;
	std::condition_variable condition;
	std::priority_queue<container, std::vector<container>, std::greater<container>> queue;

public:

	time_release_priority_queue() : alive(true) {}

	void enqueue(T* element, uint64_t priority) {
		std::unique_lock<std::mutex> lock(mutex);

		// TODO: will have to convert priority to a chrono::timepoint
		queue.push(container{element, priority});
		condition.notify_all();
	}

	bool try_dequeue(T* &element) {
		std::unique_lock<std::mutex> lock(mutex);

		if (!alive || queue.empty()) return false;

		// TODO: instead of using util::now, which converts chrono::time_point to nanoseconds,
		//       just directly use chrono::time_point
		if (queue.top().priority > util::now()) return false;

		element = queue.top().element;
		queue.pop();
		return true;
	}

	T* dequeue() {
		std::unique_lock<std::mutex> lock(mutex);

		while (alive && queue.empty()) {
			condition.wait(lock);
		}
		if (!alive) return nullptr;

		while (alive) {
			const container &top = queue.top();
			uint64_t now = util::now();

			if (top.priority <= now) break;

			
			// TODO: all of this timing should be std::chrono
			const std::chrono::nanoseconds timeout(top.priority - now);
			condition.wait_for(lock, timeout);
		}

		if (!alive) return nullptr;

		T* element = queue.top().element;
		queue.pop();
		return element;
	}

	void shutdown() {
		std::unique_lock<std::mutex> lock(mutex);
		alive = false;
		condition.notify_all();
	}
	
};
}

#endif