#ifndef _CLOCKWORK_WORKER_H_
#define _CLOCKWORK_WORKER_H_

//#include "clockwork/action_dummy.h"
//#include "clockwork/runtime_dummy.h"
#include <thread>
#include <limits>
#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "clockwork/cuda_common.h"
#include "clockwork/telemetry.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/priority_queue.h"
#include "clockwork/common.h"
#include "clockwork/task.h"
#include "clockwork/api/worker_api.h"
#include "tbb/concurrent_priority_queue.h"
#include "clockwork/memory_dummy.h"

/*
This file ties together the worker API (defined in api/worker_api.h) with model actions (defined in action.h)
using a clockwork scheduling framework (defined in runtime.h).
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
    virtual ~EngineDummy(){
        this->shutdown(false);
        delete &run_thread;
    };
    void enqueue(uint64_t end_at, std::function<void(void)> callback);
    void run();    
    void shutdown(bool await_completion){
        alive.store(false);
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
    MemoryManagerDummy* myManager;
    workerapi::Controller*  myController;

    util::GPUClockState* my_gpu_clock;//For infer's results

    //TODO CANCELL?
    ExecutorDummy(){};
    ExecutorDummy(EngineDummy* engine,  MemoryManagerDummy* manager) : available_at(util::now()),alive(true),myManager(manager),myEngine(engine){};

    void new_action(std::shared_ptr<workerapi::LoadModelFromDisk> action, uint64_t duration);
    void new_action(std::shared_ptr<workerapi::LoadWeights> action, uint64_t duration);
    void new_action(std::shared_ptr<workerapi::EvictWeights> action, uint64_t duration);
    void new_action(std::shared_ptr<workerapi::Infer> action, uint64_t duration);

    void setController(workerapi::Controller* Controller){ myController = Controller;};

    void shutdown(){alive.store(false);};

    void setGPUClock(util::GPUClockState* gpu_clock){my_gpu_clock = gpu_clock;};
};


class ClockworkRuntimeDummy {
public:
    unsigned num_gpus;
    MemoryManagerDummy* manager;// TODO WEI
    util::GPUClockState* gpu_clock;

    std::vector<EngineDummy*> engines;  // Type 3
    EngineDummy* cpu_engine;    // Type 3
    std::vector<ExecutorDummy*> gpu_executors;  // Type 3

    ExecutorDummy* load_model_executor; // Type 0
    std::vector<ExecutorDummy*> weights_executors;  // Type 1
    std::vector<ExecutorDummy*> inputs_executors;       // Type 2
    std::vector<ExecutorDummy*> outputs_executors;  // Type 4

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

    void setController(workerapi::Controller* Controller);

protected:


    void initialize(ClockworkWorkerConfig &config) {

        num_gpus = config.num_gpus;

        gpu_clock = new util::GPUClockState(num_gpus);

        manager = new MemoryManagerDummy(config);// TODO WEI

        for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            
            EngineDummy* engine = new EngineDummy();
            engines.push_back(engine);
            gpu_executors.push_back(new ExecutorDummy(engine,manager));
            gpu_executors[gpu_id]->setGPUClock(gpu_clock);
            weights_executors.push_back(new ExecutorDummy(engine,manager));
            inputs_executors.push_back(new ExecutorDummy(engine,manager));
            outputs_executors.push_back(new ExecutorDummy(engine,manager));

        }
        cpu_engine = new EngineDummy();
        load_model_executor = new ExecutorDummy(cpu_engine,manager); 

    }
};

class ClockworkDummyWorker : public workerapi::Worker {
public:
    ClockworkRuntimeDummy* runtime;// something that keeps records of gpus
    workerapi::Controller* controller;

    //Toy worker 
    ClockworkDummyWorker():runtime(new ClockworkRuntimeDummy()){};
    ClockworkDummyWorker(ClockworkWorkerConfig &config):runtime(new ClockworkRuntimeDummy(config)){};
    ~ClockworkDummyWorker(){
        this->shutdown(false);
        delete runtime;
    };
    void sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions);
    void shutdown(bool await_completion){
        runtime->shutdown(false);
        if (await_completion) {
            join();
        }
    }
    void join(){runtime->join();}

private:
    void invalidAction(std::shared_ptr<workerapi::Action> action);
    void loadModel(std::shared_ptr<workerapi::Action> action);
    void loadWeights(std::shared_ptr<workerapi::Action> action);
    void evictWeights(std::shared_ptr<workerapi::Action> action);
    void infer(std::shared_ptr<workerapi::Action> action);
    void clearCache(std::shared_ptr<workerapi::Action> action);
    void getWorkerState(std::shared_ptr<workerapi::Action> action);
};

class LoadModelFromDiskDummy{
public:
    MemoryManagerDummy* myManager;
    EngineDummy* myEngine;
    std::shared_ptr<workerapi::LoadModelFromDisk> loadmodel;
    workerapi::Controller* myController;
    uint64_t start = 0;
    uint64_t end = 0;

    LoadModelFromDiskDummy();
    LoadModelFromDiskDummy( MemoryManagerDummy* Manager,EngineDummy* Engine,std::shared_ptr<workerapi::LoadModelFromDisk> LoadModel, workerapi::Controller* Controller):myManager(Manager),myEngine(Engine),loadmodel(LoadModel), myController(Controller){};
    void run();
    void success(size_t inputs_size, size_t outputs_size,unsigned weights_pages_count, uint64_t weights_load_time_nanos,std::vector<unsigned> supported_batch_sizes,
        std::vector<uint64_t> batch_size_exec_times_nanos);
    void error(int status_code, std::string message);
};

class LoadWeightsDummy{
public:
    MemoryManagerDummy* myManager;
    EngineDummy* myEngine;
    std::shared_ptr<workerapi::LoadWeights> loadweights;
    workerapi::Controller*  myController; 
    int version;
    bool alloc_success;
    uint64_t start = 0;
    uint64_t end = 0;

    LoadWeightsDummy();
    LoadWeightsDummy( MemoryManagerDummy* Manager,EngineDummy* Engine,std::shared_ptr<workerapi::LoadWeights> LoadWeights,workerapi::Controller* Controller):myManager(Manager),myEngine(Engine),loadweights(LoadWeights), myController(Controller){version = 0; alloc_success = true;};
    void run();
    void process_completion();
    void success();
    void error(int status_code, std::string message);
};

class EvictWeightsDummy{
public:
    MemoryManagerDummy* myManager;
    EngineDummy* myEngine;
    std::shared_ptr<workerapi::EvictWeights> evictweights;
    workerapi::Controller*  myController;
    uint64_t start = 0;
    uint64_t end = 0;

    EvictWeightsDummy();
    EvictWeightsDummy( MemoryManagerDummy* Manager,EngineDummy* Engine,std::shared_ptr<workerapi::EvictWeights> EvictWeights, workerapi::Controller* Controller):myManager(Manager),myEngine(Engine),evictweights(EvictWeights), myController(Controller){};
    void run();
    void success();
    void error(int status_code, std::string message);
};

class InferDummy{
public:
    MemoryManagerDummy* myManager;
    EngineDummy* myEngine;
    std::shared_ptr<workerapi::Infer> infer;
    workerapi::Controller*  myController;

    unsigned gpu_clock_before;
    util::GPUClockState* gpu_clock;
    int version; 
    uint64_t start = 0;
    uint64_t end = 0;

    InferDummy();
    InferDummy( MemoryManagerDummy* Manager,EngineDummy* Engine,std::shared_ptr<workerapi::Infer> Infer,workerapi::Controller* Controller, util::GPUClockState* gpu_clock):myManager(Manager),myEngine(Engine),infer(Infer), myController(Controller),gpu_clock(gpu_clock){version = 0;};
    void run();
    void process_completion();
    void success();
    void error(int status_code, std::string message);
};


uint64_t adjust_timestamp(uint64_t timestamp, int64_t clock_delta);
}

#endif
