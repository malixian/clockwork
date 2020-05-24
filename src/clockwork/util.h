#ifndef _CLOCKWORK_UTIL_H_
#define _CLOCKWORK_UTIL_H_

#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <deque>

/* These two files are included for the Order Statistics Tree. */
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

#define NUM_GPUS_1 1
#define NUM_GPUS_2 2
#define GPU_ID_0 0

namespace clockwork {

typedef std::chrono::steady_clock::time_point time_point;

namespace util {

// High-resolution timer, current time in nanoseconds
std::uint64_t now();
std::string millis(uint64_t t);

time_point hrt();

std::uint64_t nanos(time_point t);

std::string nowString();

unsigned get_num_gpus();

void setCudaFlags();

std::string getGPUmodel(int deviceNumber);

extern "C" char* getGPUModelToBuffer(int deviceNumber, char* buf);

void printCudaVersion();


void readFileAsString(const std::string &filename, std::string &dst);
std::vector<std::string> listdir(std::string directory);
bool exists(std::string filename);
long filesize(std::string filename);


void initializeCudaStream(unsigned gpu_id = 0, int priority = 0);
void SetStream(cudaStream_t stream);
cudaStream_t Stream();

// A hash function used to hash a pair of any kind
// Source: https://www.geeksforgeeks.org/how-to-create-an-unordered_map-of-pairs-in-c/
struct hash_pair {
	template <class T1, class T2>
	size_t operator()(const std::pair<T1, T2>& p) const {
		auto hash1 = std::hash<T1>{}(p.first);
		auto hash2 = std::hash<T2>{}(p.second);
		return hash1 ^ hash2;
	}
};

struct hash_tuple {
	template <class T1, class T2, class T3>
	size_t operator()(const std::tuple<T1, T2, T3>& t) const {
		auto hash1 = std::hash<T1>{}(std::get<0>(t) );
		auto hash2 = std::hash<T2>{}(std::get<1>(t) );
		auto hash3 = std::hash<T3>{}(std::get<2>(t) );
		return hash1 ^ hash2 ^ hash3;
	}
};

std::string get_clockwork_directory();

std::string get_example_model_path(std::string model_name = "resnet18_tesla-m40");

std::string get_example_model_path(std::string clockwork_directory, std::string model_name);

std::string get_modelzoo_dir();
std::string get_clockwork_model(std::string shortname);

std::map<std::string, std::string> get_clockwork_modelzoo();

class GPUClockState {
 private:
  	std::atomic_bool alive = true;
 	std::thread checker;
 	std::vector<unsigned> clock;

 public:
 	GPUClockState(unsigned num_gpus);

 	void run();
 	void shutdown();
 	void join();

 	unsigned get(unsigned gpu_id);

};

template <typename T> 
class SlidingWindowT {
private:
	unsigned window_size;

	/* An order statistics tree is used to implement a wrapper around a C++
	   set with the ability to know the ordinal number of an item in the set
	   and also to get an item by its ordinal number from the set.
	   The data structure I use is implemented in STL but only for GNU C++.
	   Some sources are documented below:
	     -- https://gcc.gnu.org/onlinedocs/libstdc++/ext/pb_ds/
	     -- https://codeforces.com/blog/entry/11080
	     -- https://gcc.gnu.org/onlinedocs/libstdc++/ext/pb_ds/tree_based_containers.html
	     -- https://opensource.apple.com/source/llvmgcc42/llvmgcc42-2336.9/libstdc++-v3/testsuite/ext/pb_ds/example/tree_order_statistics.cc.auto.html
		 -- https://stackoverflow.com/questions/44238144/order-statistics-tree-using-gnu-pbds-for-multiset
	     -- https://www.geeksforgeeks.org/order-statistic-tree-using-fenwick-tree-bit/ */

	typedef __gnu_pbds::tree<
		T,
		__gnu_pbds::null_type,
		std::less_equal<T>,
		__gnu_pbds::rb_tree_tag,
		__gnu_pbds::tree_order_statistics_node_update> OrderedMultiset;

	/* We maintain a list of data items (FIFO ordered) so that the latest
	   and the oldest items can be easily tracked for insertion and removal.
	   And we also maintain a parallel OrderedMultiset data structure where the
	   items are stored in an order statistics tree so that querying, say, the
	   99th percentile value is easy. We also maintain an upper bound on sliding
	   window size. After the first few iterations, the number of data items
	   is always equal to the upper bound. Thus, we have:
			-- Invariant 1: q.size() == oms.size()
			-- Invariant 2: q.size() <= window_size */
	std::list<T> q;
	OrderedMultiset oms;

public:
	SlidingWindowT() : window_size(100) {}
	SlidingWindowT(unsigned window_size) : window_size(window_size) {}

	/* Assumption: q.size() == oms.size() */
	unsigned get_size() { return q.size(); }

	/* Requirement: rank < oms.size() */
	T get_value(unsigned rank) { return (*(oms.find_by_order(rank))); }
	T get_percentile(float percentile) {
		float position = percentile * (q.size() - 1);
		unsigned up = ceil(position);
		unsigned down = floor(position);
		if (up == down) return get_value(up);
		return get_value(up) * (position - down) + get_value(down) * (up - position);
	}
	T get_min() { return get_value(0); }
	T get_max() { return get_value(q.size()-1); }
	void insert(T latest) {
		q.push_back(latest);
		oms.insert(latest);
		if (q.size() > window_size) {
			uint64_t oldest = q.front();
			q.pop_front();
			auto it=oms.upper_bound (oldest);
			oms.erase(it); // Assumption: *it == oldest
		}
	}
	
	SlidingWindowT(unsigned window_size, T initial_value) : window_size(window_size) {
		insert(initial_value);
	}
};

class SlidingWindow : public SlidingWindowT<uint64_t> {

public:
	SlidingWindow() : SlidingWindowT(100) {}
	SlidingWindow(unsigned window_size) : SlidingWindowT(window_size) {}
};



class WorkerTracker {
private:
	struct Work { int id; uint64_t size; uint64_t work_begin; };
	int clock_;
	uint64_t work_begin = 0;
	uint64_t lag; // allowable lag
	uint64_t future; // time into the future to schedule
	std::deque<Work> outstanding;
	uint64_t total_outstanding = 0;

	void update(int id, uint64_t time_of_completion, bool success) {
		if (outstanding.front().id == id) {
			total_outstanding -= outstanding.front().size;
			work_begin = time_of_completion;
			outstanding.pop_front();
		} else {
			// the work was done but we don't know when
			for (auto it = outstanding.begin(); it != outstanding.end(); it++){
				if (it->id == id) {
					total_outstanding -= it->size;
					if (success) work_begin += it->size/clock_;
					outstanding.erase(it);
					break;
				}
			}
		}
		if (outstanding.size() > 0) {
			work_begin = std::max(work_begin, outstanding.front().work_begin);
		}
	}

public:

	WorkerTracker(int clock, uint64_t lag = 100000000UL, uint64_t future=0UL) : clock_(clock), lag(lag), future(future) {}

	// Returns the time outstanding work will complete
	uint64_t available() {
		uint64_t now = util::now();
		uint64_t work_begin = this->work_begin;
		if (outstanding.size() > 0 && (work_begin + (outstanding.front().size / clock_) + lag < now)) {
			// Outstanding work has mysteriously not completed
			work_begin = now - lag - outstanding.front().size / clock_;
		}
		return std::max(work_begin + total_outstanding / clock_, now + future);
	}

	int clock() {
		return clock_;
	}

	void update_clock(int clock) {
		this->clock_ = clock;
	}

	void add(int id, uint64_t work_size, uint64_t work_begin = 0) {
		if (outstanding.empty()) {
			this->work_begin = std::max(work_begin, now());
		}
		uint64_t size = work_size * clock_;
		outstanding.push_back({id, size, work_begin});
		total_outstanding += size;
	}

	void success(int id, uint64_t time_of_completion) {
		update(id, time_of_completion, true);
	}

	void error(int id, uint64_t time_of_completion) {
		update(id, time_of_completion, false);
	}

};

std::vector<unsigned> make_batch_lookup(std::vector<unsigned> supported_batch_sizes);


#define DEBUG_PRINT(msg) \
	std::cout << __FILE__ << "::" << __LINE__ << "::" << __FUNCTION__ << " "; \
	std::cout << msg << std::endl;

}
}


#endif
