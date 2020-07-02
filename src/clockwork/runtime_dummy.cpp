#include "clockwork/api/worker_api.h"
#include "clockwork/runtime_dummy.h"
#include "clockwork/action_dummy.h"
#include "clockwork/thread.h"

namespace clockwork {


void EngineDummy::enqueue(uint64_t end_at, std::function<void(void)> callback){
	if (end_at - util::now() <= 0) callback();
	else pending_actions.push(element{end_at, callback});
}

void EngineDummy::run() {
    while (alive.load()) {
        element next;
        while (pending_actions.try_pop(next)){
        	if(util::now() > next.ready) next.callback();
        	else{
        		//std::this_thread::sleep_until(next.ready); //wei TODOD
        		usleep(1000);
        		next.callback();
        	}
        }
    }
    shutdown(true);
}

void ExecutorDummy::new_action(std::shared_ptr<workerapi::LoadModelFromDisk> action, uint64_t duration){
	if(alive){
        LoadWeightsAction* loadweights = new LoadWeightsActionDummy(myRuntime,action);
		uint64_t start_at = std::max(util::now(), available_at);
    	uint64_t end_at = start_at + duration;
		if(start_at > action->latest)
			myEngine->enqueue(end_at, action.onError("couldn't execute in time"))
		else if(start_at < action->earlist)
            myEngine->enqueue(end_at, action.onError("ran before it was eligible"))
        else{
			available_at = end_at
			myEngine->enqueue(end_at, action.onComplete)
		}
	}
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::LoadWeights> action, uint64_t duration){
    if(alive){
        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at > action->latest)
            myEngine->enqueue(end_at, action.onError("couldn't execute in time"))
        else{
            available_at = end_at
            myEngine->enqueue(end_at, action.onComplete)
        }
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::EvictWeights> action, uint64_t duration){
    if(alive){
        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at > action->latest)
            myEngine->enqueue(end_at, action.onError("couldn't execute in time"))
        else{
            available_at = end_at
            myEngine->enqueue(end_at, action.onComplete)
        }
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::Infer> action, uint64_t duration){
    if(alive){
        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at > action->latest)
            myEngine->enqueue(end_at, action.onError("couldn't execute in time"))
        else{
            available_at = end_at
            myEngine->enqueue(end_at, action.onComplete)
        }
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::ClearCache> action){
    if(alive){
        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at > action->latest)
            myEngine->enqueue(end_at, action.onError("couldn't execute in time"))
        else{
            available_at = end_at
            myEngine->enqueue(end_at, action.onComplete)
        }
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::GetWorkerState> action){
    if(alive){
        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at > action->latest)
            myEngine->enqueue(end_at, action.onError("couldn't execute in time"))
        else{
            available_at = end_at
            myEngine->enqueue(end_at, action.onComplete)
        }
    }
};

void ClockworkRuntimeDummy::shutdown(bool await_completion) {
	/* 
	Stop executors.  They'll finish current tasks, prevent enqueueing
	new tasks, and cancel tasks that haven't been started yet
	*/
	load_model_executor->shutdown();
	cpu_engine->shutdown(await_completion);
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		gpu_executors[gpu_id]->shutdown();
		weights_executors[gpu_id]->shutdown();
		inputs_executors[gpu_id]->shutdown();
		outputs_executors[gpu_id]->shutdown();
		engines[gpu_id]->shutdown(await_completion);
	}
		if (await_completion) {
		join();
	}
}

void ClockworkRuntimeDummy::join() {
	/*
	Wait for executors to be finished
	*/
	cpu_engine->join();
	for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		engines[gpu_id]->join();
	}
}


}
