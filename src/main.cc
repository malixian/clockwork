#include <iostream>
#include "clockwork/queue.h"
#include "tbb/task_scheduler_init.h"
#include "clockwork/runtime.h"
#include "clockwork/threadpoolruntime.h"
#include <sstream>
#include <atomic>

using namespace clockwork;

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

	Runtime* runtime = newFIFOThreadpoolRuntime(4);

	int expected = 0;
	std::atomic_int* actual = new std::atomic_int{0};
	for (unsigned requestID = 1; requestID < 10; requestID++) {
		RequestBuilder* b = runtime->newRequest();
		for (unsigned taskID = 0; taskID < requestID; taskID++) {
			expected++;
			b = b->addTask(TaskType::CPU, [=] {
				std::stringstream ss;
				ss << "request-" << requestID << "   task-" << taskID << std::endl;
				std::cout << ss.str();
				actual->fetch_add(1);
			});
		}
		b->submit();
	}


	while (actual->load() < expected) {}

	runtime->shutdown(true);
	delete actual;

	std::cout << "end" << std::endl;
}
