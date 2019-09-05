#include <iostream>
#include "clockwork/queue.h"
#include "tbb/task_scheduler_init.h"
#include "clockwork/runtime.h"
#include "clockwork/threadpoolruntime.h"

using namespace clockwork;

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

	Runtime* runtime = newFIFOThreadpoolRuntime(4);

	std::cout << "end" << std::endl;
}
