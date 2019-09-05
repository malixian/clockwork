#include <iostream>
#include "clockwork/clockwork_queue.h"
#include <clockwork/manager.h>

using namespace clockwork;

Manager manager;

int main(int argc, char *argv[]) {

	std::cout << "wwhaa\n";

	manager.loadModel("testboi", "/home/projects/modelzoo/resnet50/tesla-m40_batchsize2/tvm_model");

	std::cout << "Hello world" << std::endl;

	PriorityQueue::saysomething();
}
