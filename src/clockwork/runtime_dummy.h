#ifndef _CLOCKWORK_RUNTIME_H_
#define _CLOCKWORK_RUNTIME_H_

#include <thread>
#include <limits>
#include <algorithm>
#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/cuda_common.h"
#include "clockwork/telemetry.h"
#include "../src/clockwork/telemetry/task_telemetry_logger.h"
#include "../src/clockwork/telemetry/action_telemetry_logger.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/priority_queue.h"
#include "clockwork/common.h"
#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_priority_queue.h"
#include "clockwork/task.h"
#include "clockwork/memory.h"
#include "clockwork/config.h"

/*
This file contains the clockwork scheduling and thread pool logic for executing tasks, asynchronous
tasks, and checking async task completion.
*/

namespace clockwork {

class ClockworkRuntimeDummy;

class EngineDummy{

private:
    struct element {
        uint64_t ready;
        std::function<void(void)> callback;

        friend bool operator < (const element& lhs, const element &rhs) {
            return lhs.ready < rhs.ready;
        }
        friend bool operator > (const element& lhs, const element &rhs) {
            return lhs.ready > rhs.ready;
        }
    };

public:
    std::thread run_thread;
    tbb::concurrent_priority_queue<element, std::greater<element>> pending_actions;
    std::atomic_bool alive;
    EngineDummy(): run_thread(&EngineDummy::run, this),alive(true){};
    ~EngineDummy();
    void enqueue(uint64_t end_at, std::function<void(void)> callback);
    void run();    
    void shutdown(bool await_completion){
        alive = false;
        if (await_completion) {
            join();
        }
    }
    void join(){run_thread.join();}

};

class ExecutorDummy{
public:
	uint64_t available_at;
	EngineDummy* myEngine;
	std::atomic_bool alive;
    ClockworkRuntimeDummy* myRuntime;


	ExecutorDummy(){}
	ExecutorDummy(EngineDummy* engine, ClockworkRuntimeDummy* runtime) : available_at(util::now()),alive(true),myRuntime(runtime),myEngine(engine){};

	void new_action(std::shared_ptr<workerapi::LoadModelFromDisk> action, uint64_t duration);
	void new_action(std::shared_ptr<workerapi::LoadWeights> action, uint64_t duration);
	void new_action(std::shared_ptr<workerapi::EvictWeights> action, uint64_t duration);
	void new_action(std::shared_ptr<workerapi::Infer> action, uint64_t duration);
	void new_action(std::shared_ptr<workerapi::ClearCache> action);
	void new_action(std::shared_ptr<workerapi::GetWorkerState> action);

	void shutdown(){alive = false;};
};


class ClockworkRuntimeDummy {
public:
	unsigned num_gpus;
	MemoryManager* manager;// TODO WEI
	util::GPUClockState* gpu_clock;

	std::vector<EngineDummy*> engines;	// Type 3
	EngineDummy* cpu_engine;	// Type 3
	std::vector<ExecutorDummy*> gpu_executors;	// Type 3

	ExecutorDummy* load_model_executor;	// Type 0
	std::vector<ExecutorDummy*> weights_executors;	// Type 1
	std::vector<ExecutorDummy*> inputs_executors;		// Type 2
	std::vector<ExecutorDummy*> outputs_executors;	// Type 4

	ClockworkRuntimeDummy() {
		ClockworkWorkerConfig config;
		initialize(config);
	}

	ClockworkRuntimeDummy(ClockworkWorkerConfig &config) {
		initialize(config);
	}

	virtual ~ClockworkRuntimeDummy() {
		delete manager;
		delete load_model_executor;
		delete cpu_engine;

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			delete gpu_executors[gpu_id];
			delete weights_executors[gpu_id];
			delete inputs_executors[gpu_id];
			delete outputs_executors[gpu_id];
			delete engines[gpu_id];

		}
	}

	void shutdown(bool await_completion);

	void join();

protected:


	void initialize(ClockworkWorkerConfig &config) {

		num_gpus = config.num_gpus;

		gpu_clock = new util::GPUClockState(num_gpus);

		manager = new MemoryManager(config);// TODO WEI

		for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
			
			EngineDummy* engine = new EngineDummy();
			engines.push_back(engine);
			gpu_executors.push_back(new ExecutorDummy(engine,this));
			weights_executors.push_back(new ExecutorDummy(engine,this));
			inputs_executors.push_back(new ExecutorDummy(engine,this));
			outputs_executors.push_back(new ExecutorDummy(engine,this));

		}

		cpu_engine = new EngineDummy();
		load_model_executor = new ExecutorDummy(cpu_engine,this); 

		std::string task_file_path = config.telemetry_log_dir + "/" + config.task_telemetry_log_file;
		std::string action_file_path = config.telemetry_log_dir + "/" + config.action_telemetry_log_file;

	}
};


}

#endif
