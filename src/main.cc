#include <iostream>
#include "clockwork/queue.h"
#include "tbb/task_scheduler_init.h"
#include "clockwork/util.h"

using namespace clockwork;

class Task : public Queueable {
public:
	int id;
	uint64_t created;
	Task(int id) : id(id), created(util::now()) {}
	virtual bool isComplete() {
		return util::now() - created > 1000000000;
	}
};

uint64_t stuff() {
	uint64_t begin = util::now();

	MaxPriorityQueue<Task, 1> q;

	q.enqueue(Task(4), 4);
	q.enqueue(Task(1), 1);
	q.enqueue(Task(3), 3);
	q.enqueue(Task(2), 2);

	std::cout << q.dequeue().id << std::endl;
	std::cout << q.dequeue().id << std::endl;
	std::cout << q.dequeue().id << std::endl;
	std::cout << q.dequeue().id << std::endl;

	uint64_t end = util::now();

	std::cout << "done" << std::endl;
	return end-begin;
}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

	std::vector<uint64_t> runtimes;

	for (unsigned i = 0; i < 10; i++) {
		runtimes.push_back(stuff());
	}

	for (unsigned i = 0; i < runtimes.size(); i++) {
		std::cout << i << ": " << runtimes[i] << std::endl;
	}
}
