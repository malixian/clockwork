#include "clockwork/worker_dummy.h"
#include <dmlc/logging.h>
#include <algorithm>
#include "clockwork/modeldef.h"

#include <iostream>

namespace clockwork {

EngineDummy::EngineDummy(unsigned num_gpus){
    for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++){
        infers_to_end.push_back(nullptr);
        loads_to_end.push_back(nullptr);
    }
}

void EngineDummy::addExecutor(ExecutorDummy* executor){executors.push_back(executor);}

void EngineDummy::addToEnd(int type, unsigned gpu_id, element* action){
    if(type == workerapi::loadWeightsAction)
        loads_to_end[gpu_id] = action;
    else if (type == workerapi::loadWeightsAction)
        infers_to_end[gpu_id] = action;
}

void EngineDummy::startEngine(){
    alive.store(true);
    run_thread = std::thread(&EngineDummy::run, this);
}

void EngineDummy::run() {
    while(alive.load()){
        uint64_t timestamp = util::now();
        element next;
        for(ExecutorDummy* executor: executors){
            if(executor->type == workerapi::loadWeightsAction){
                if(loads_to_end[executor->gpu_id] != nullptr){
                    if(loads_to_end[executor->gpu_id]->ready <= timestamp){
                        loads_to_end[executor->gpu_id]->callback();
                        loads_to_end[executor->gpu_id] = nullptr;
                    }
                }else{
                    if(executor->actions_to_start.try_pop(next))
                        next.callback();//loads_to_end[executor->gpu_id] = &element{end_at,loadweights_on_complete} if on_start succeed
                }
            }else if(executor->type == workerapi::inferAction){
                if(infers_to_end[executor->gpu_id] != nullptr){
                    if(infers_to_end[executor->gpu_id]->ready <= timestamp){
                        infers_to_end[executor->gpu_id]->callback();
                        infers_to_end[executor->gpu_id] = nullptr;
                    }
                }else{
                    if(executor->actions_to_start.try_pop(next))
                        next.callback();
                }
            }else if(executor->actions_to_start.try_pop(next)){
                next.callback();
            } 
        }
    }
    shutdown(true);
}

void ExecutorDummy::new_action(std::shared_ptr<workerapi::LoadModelFromDisk> action){
    if(alive.load()){
        LoadModelFromDiskDummy* loadmodel = new LoadModelFromDiskDummy(myManager,myEngine,action,myController);
        actions_to_start.push(element{loadmodel->loadmodel->earliest, [loadmodel]() {loadmodel->run();} });
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::LoadWeights> action){
    if(alive.load()){
        LoadWeightsDummy* loadweights = new LoadWeightsDummy(myManager,myEngine,action, myController);
        actions_to_start.push(element{loadweights->loadweights->earliest,[loadweights]() {loadweights->run();} });
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::EvictWeights> action){
    if(alive.load()){
        EvictWeightsDummy* evictweights = new EvictWeightsDummy(myManager,myEngine,action, myController);
        actions_to_start.push(element{evictweights->evictweights->earliest, [evictweights]() {evictweights->run();} });
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::Infer> action){
    if(alive.load()){
        InferDummy* infer = new InferDummy(myManager,myEngine,action, myController);
        actions_to_start.push(element{infer->infer->earliest, [infer]() {infer->run();} });
    }
};

void ClockworkRuntimeDummy::setController(workerapi::Controller* Controller){
    for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            gpu_executors[gpu_id]->setController(Controller);
            weights_executors[gpu_id]->setController(Controller);
            outputs_executors[gpu_id]->setController(Controller);
    }
    load_model_executor->setController(Controller);
}

void ClockworkRuntimeDummy::shutdown(bool await_completion) {
    /* 
    Stop executors.  They'll finish current tasks, prevent enqueueing
    new tasks, and cancel tasks that haven't been started yet
    */
    engine->shutdown(await_completion);
    load_model_executor->shutdown(); 
    for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        gpu_executors[gpu_id]->shutdown();
        weights_executors[gpu_id]->shutdown();
        outputs_executors[gpu_id]->shutdown();
    }
    if (await_completion) {
        join();
    }
}

void ClockworkRuntimeDummy::join() {
    /*
    Wait for executors to be finished
    */
    engine->join();
}

void ClockworkDummyWorker::sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions) {
    for (std::shared_ptr<workerapi::Action> action : actions) {
        switch (action->action_type) {
            case workerapi::loadModelFromDiskAction: loadModel(action); break;
            case workerapi::loadWeightsAction: loadWeights(action); break;
            case workerapi::inferAction: infer(action); break;
            case workerapi::evictWeightsAction: evictWeights(action); break;
            case workerapi::clearCacheAction: clearCache(action); break;
            case workerapi::getWorkerStateAction: getWorkerState(action); break;
            default: invalidAction(action); break;
        }
    }
}

void ClockworkDummyWorker::invalidAction(std::shared_ptr<workerapi::Action> action) {
    auto result = std::make_shared<workerapi::ErrorResult>();

    result->id = action->id;
    result->action_type = action->action_type;
    result->status = actionErrorRuntimeError;
    result->message = "Invalid Action";

    controller->sendResult(result);
}

// Need to be careful of timestamp = 0 and timestamp = UINT64_MAX which occur often
// and clock_delta can be positive or negative
uint64_t adjust_timestamp(uint64_t timestamp, int64_t clock_delta) {
    if (clock_delta >= 0) return std::max(timestamp, timestamp + clock_delta);
    else return std::min(timestamp, timestamp + clock_delta);
}

void ClockworkDummyWorker::loadModel(std::shared_ptr<workerapi::Action> action) {
    auto load_model = std::static_pointer_cast<workerapi::LoadModelFromDisk>(action);
    if (load_model != nullptr) {
        // It is a hack to do this here, but easiest / safest place to do it for now
        load_model->earliest = adjust_timestamp(load_model->earliest, load_model->clock_delta);
        load_model->latest = adjust_timestamp(load_model->latest, load_model->clock_delta);

        runtime->load_model_executor->new_action(load_model);
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::loadWeights(std::shared_ptr<workerapi::Action> action) {
    auto load_weights = std::static_pointer_cast<workerapi::LoadWeights>(action);
    if (load_weights != nullptr) {
        // It is a hack to do this here, but easiest / safest place to do it for now
        load_weights->earliest = adjust_timestamp(load_weights->earliest, load_weights->clock_delta);
        load_weights->latest = adjust_timestamp(load_weights->latest, load_weights->clock_delta);

        runtime->weights_executors[load_weights->gpu_id]->new_action(load_weights);      
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::evictWeights(std::shared_ptr<workerapi::Action> action) {
    auto evict_weights = std::static_pointer_cast<workerapi::EvictWeights>(action);
    if (evict_weights != nullptr) {
        // It is a hack to do this here, but easiest / safest place to do it for now
        evict_weights->earliest = adjust_timestamp(evict_weights->earliest, evict_weights->clock_delta);
        evict_weights->latest = adjust_timestamp(evict_weights->latest, evict_weights->clock_delta);

        runtime->outputs_executors[evict_weights->gpu_id]->new_action(evict_weights);
        
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::infer(std::shared_ptr<workerapi::Action> action) {
    auto infer = std::static_pointer_cast<workerapi::Infer>(action);
    if (infer != nullptr) {
        // It is a hack to do this here, but easiest / safest place to do it for now
        infer->earliest = adjust_timestamp(infer->earliest, infer->clock_delta);
        infer->latest = adjust_timestamp(infer->latest, infer->clock_delta);

        runtime->gpu_executors[infer->gpu_id]->new_action(infer);
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::clearCache(std::shared_ptr<workerapi::Action> action) {
    auto clear_cache = std::static_pointer_cast<workerapi::ClearCache>(action);
    if (clear_cache != nullptr) {
        runtime->manager->models->clearWeights();
        for (unsigned i = 0; i < runtime->num_gpus; i++) {
            runtime->manager->weights_caches[i]->clear();
        }
        auto result = std::make_shared<workerapi::ClearCacheResult>();
        result->id = action->id;
        result->action_type = workerapi::clearCacheAction;
        result->status = actionSuccess; 
        controller->sendResult(result);
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::getWorkerState(std::shared_ptr<workerapi::Action> action) {
    auto get_worker_state = std::static_pointer_cast<workerapi::GetWorkerState>(action);
    if (get_worker_state != nullptr) {
        auto result = std::make_shared<workerapi::GetWorkerStateResult>();
        result->id = action->id;
        result->action_type = workerapi::getWorkerStateAction;
        runtime->manager->get_worker_memory_info(result->worker);
        result->status = actionSuccess; 
        controller->sendResult(result);
    } else {
        invalidAction(action);
    }
}

void LoadModelFromDiskDummy::error(int status_code, std::string message){
    TaskError* error = new TaskError(status_code,message);
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->id = loadmodel->id;
    result->action_type = workerapi::loadModelFromDiskAction;
    result->status = error->status_code;
    result->message = error->message;
    result->action_received = adjust_timestamp(loadmodel->received, -loadmodel->clock_delta);
    result->clock_delta = loadmodel->clock_delta;
    myController->sendResult(result);
    delete this;
}

void LoadModelFromDiskDummy::success(size_t inputs_size, size_t outputs_size,unsigned weights_pages_count, uint64_t weights_load_time_nanos,std::vector<unsigned> supported_batch_sizes,
        std::vector<uint64_t> batch_size_exec_times_nanos) {

    auto result = std::make_shared<workerapi::LoadModelFromDiskResult>();

    result->id = loadmodel->id;
    result->action_type = workerapi::loadModelFromDiskAction;
    result->status = actionSuccess;
    result->input_size = inputs_size;
    result->output_size = outputs_size;
    result->copies_created = loadmodel->no_of_copies;
    result->weights_load_time_nanos = weights_load_time_nanos;
    result->supported_batch_sizes = supported_batch_sizes;
    result->batch_size_exec_times_nanos = batch_size_exec_times_nanos;

    // TODO Verify: I assume that GPU-specific weights_caches have identical page_size
    size_t page_size = myManager->weights_caches[0]->page_size;
    result->num_weights_pages = weights_pages_count;

    result->weights_size_in_cache = result->num_weights_pages * page_size;

    //Set timestamps in the result
    result->begin = adjust_timestamp(start, -loadmodel->clock_delta);
    result->end = adjust_timestamp(end, -loadmodel->clock_delta);
    result->duration = result->end - result->begin;
    result->action_received = adjust_timestamp(loadmodel->received, -loadmodel->clock_delta);
    result->clock_delta = loadmodel->clock_delta;

    myController->sendResult(result);
    delete this;
}

void LoadModelFromDiskDummy::run(){

    //Check if model is already loaded
    start = util::now();
    std::stringstream err;
    if(start < loadmodel->earliest){
        err << "LoadModelFromDiskTask ran before it was eligible"
            << " (now " << util::millis(start)
            << ", earliest " << util::millis(loadmodel->earliest) << ")";
        error(actionErrorRuntimeError,err.str());
        return;

    }else if(start > loadmodel->latest){
        err << "LoadModelFromDiskTask could not start in time"
            << " (now " << util::millis(start)
            << ", latest " << util::millis(loadmodel->latest) << ")";
        error(actionErrorCouldNotStartInTime, err.str());
        return;
    }

    std::vector<unsigned> gpu_ids;
    for (unsigned gpu_id = 0; gpu_id < myManager->num_gpus; gpu_id++) {
        gpu_ids.push_back(gpu_id);
        for (unsigned i = 0; i < loadmodel->no_of_copies; i++) {
            if (myManager->models->contains(loadmodel->model_id+i, gpu_id)) {
                error(actionErrorInvalidModelID, "LoadModelFromDiskTask specified ID that already exists");
                return;
            }
        }
    }

    
    try{
        // Load data for batch sizes and extract performance profile
        std::vector<unsigned> supported_batch_sizes;
        std::vector<uint64_t> batch_size_exec_times_nanos;
        uint64_t weights_load_time_nanos;

        std::vector<ModelDataDummy> modeldata = loadModelDataDummy(loadmodel->model_path);
        weights_load_time_nanos = modeldata[0].weights_measurement;
        for (ModelDataDummy &d : modeldata) {
                if (d.batch_size <= loadmodel->max_batch_size && 
                    (d.batch_size == 1 || d.exec_measurement <= loadmodel->max_exec_duration)) {
                    supported_batch_sizes.push_back(d.batch_size);
                    batch_size_exec_times_nanos.push_back(d.exec_measurement);        
                }
        }

        //deserialize the model metadata
        model::PageMappedModelDef* spec = new model::PageMappedModelDef();

        model::PageMappedModelDef::ReadFrom(modeldata[0].serialized_spec, *spec);
        CHECK(spec != nullptr) << " spec is nullptr";

        //Extract model metadata
        unsigned weights_pages_count = spec->weights_pages.size();
        uint64_t weights_size = weights_pages_count*spec->configured_weights_page_size;
        size_t inputs_size = 0;
        size_t outputs_size = 0;
    
        for (auto &input : spec->inputs) {
            inputs_size += input.size;
        }

        for (auto &output : spec->outputs) {
            outputs_size += output.size;
        }

        //Add model to modelstore
        for (auto &gpu_id : gpu_ids) {
            for (unsigned i = 0; i < loadmodel->no_of_copies; i++) {
                workerapi::ModelInfo* modelInfo = new workerapi::ModelInfo();
                modelInfo->id = loadmodel->model_id + i;
                modelInfo->source = loadmodel->model_path;
                modelInfo->input_size = inputs_size;
                modelInfo->output_size = outputs_size;
                modelInfo->supported_batch_sizes = supported_batch_sizes;
                modelInfo->weights_size = weights_size;
                modelInfo->num_weights_pages = spec->configured_weights_page_size;
                modelInfo->weights_load_time_nanos = weights_load_time_nanos;
                modelInfo->batch_size_exec_times_nanos = batch_size_exec_times_nanos;
                RuntimeModelDummy* rm = new RuntimeModelDummy(modelInfo,gpu_id,weights_pages_count);

                bool success = myManager->models->put_if_absent(
                    loadmodel->model_id + i, 
                    gpu_id, 
                    rm
                );
                CHECK(success) << "Loaded models changed while loading from disk";
            }
        }

        end = util::now();
        success(inputs_size,outputs_size,weights_pages_count,weights_load_time_nanos,supported_batch_sizes,batch_size_exec_times_nanos);

    }catch (dmlc::Error &errMessage) {
        error(actionErrorInvalidModelID, errMessage.what());
        return;
    }catch(NoMeasureFile &errMessage){
        error(errMessage.status_code, errMessage.message);
        return;
    }
    

}



void LoadWeightsDummy::run(){
    start = util::now();

    std::stringstream err;
    if(start < loadweights->earliest){
        err << "LoadWeights ran before it was eligible"
            << " (now " << util::millis(start)
            << ", earliest " << util::millis(loadweights->earliest) << ")";
        error(actionErrorRuntimeError, err.str());
        return;

    }else if(start > loadweights->latest){
        err << "LoadWeights could not start in time"
            << " (now " << util::millis(start)
            << ", latest " << util::millis(loadweights->latest) << ")";
        error(actionErrorCouldNotStartInTime, err.str());
        return;
    }

    RuntimeModelDummy* rm = myManager->models->get(loadweights->model_id, loadweights->gpu_id);
    if (rm == nullptr) {
        std::string message = "LoadWeightsTask could not find model";
        message += " with model ID " + std::to_string(loadweights->model_id);
        message += " and GPU ID " + std::to_string(loadweights->gpu_id);
        error(loadWeightsUnknownModel, message);
        return;
    }
    rm->lock();
    if (!rm->weights) {
        alloc_success = myManager->weights_caches[loadweights->gpu_id]->alloc(rm->weightspagescount);
    }
    version = ++rm->version;
    rm->unlock();
    end = start + rm->modelinfo->weights_load_time_nanos;
    element* action = new element();
    action->ready = end;
    action->callback = [this]() {this->process_completion();};
    myEngine->addToEnd(workerapi::loadWeightsAction, loadweights->gpu_id,action);
}

void LoadWeightsDummy::process_completion(){
    RuntimeModelDummy* rm = myManager->models->get(loadweights->model_id, loadweights->gpu_id);
    bool version_unchanged = false;
    rm->lock();
    if (rm->version == version) {
        version_unchanged = true;
    }
    rm->unlock();
    if (version_unchanged) {
        if(alloc_success){
            rm->lock();
            rm->weights = true;
            rm->unlock();
            success();
        }else{
            error(loadWeightsInsufficientCache, "LoadWeightsTask failed to allocate pages from cache");
        }
    }else {
        error(loadWeightsConcurrentModification, "Model weights were modified while being copied");
    }
}

void LoadWeightsDummy::success(){
    auto result = std::make_shared<workerapi::LoadWeightsResult>();

    result->id = loadweights->id;
    result->action_type = workerapi::loadWeightsAction;
    result->status = actionSuccess;

    //Set timestamps in the result
    result->begin = adjust_timestamp(start, -loadweights->clock_delta);
    result->end = adjust_timestamp(end, -loadweights->clock_delta);
    result->duration = result->end - result->begin;
    result->action_received = adjust_timestamp(loadweights->received, -loadweights->clock_delta);
    result->clock_delta = loadweights->clock_delta;
    
    myController->sendResult(result);
    delete this;
}

void LoadWeightsDummy::error(int status_code, std::string message){
    TaskError* error = new TaskError(status_code,message);
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->id = loadweights->id;
    result->action_type = workerapi::loadWeightsAction;
    result->status = error->status_code;
    result->message = error->message;
    result->action_received = adjust_timestamp(loadweights->received, -loadweights->clock_delta);
    result->clock_delta = loadweights->clock_delta;
    myController->sendResult(result);
    delete this;
}

void EvictWeightsDummy::run(){
    start = util::now();

    std::stringstream err;
    if(start < evictweights->earliest){
        err << "EvictWeights ran before it was eligible"
            << " (now " << util::millis(start)
            << ", earliest " << util::millis(evictweights->earliest) << ")";
        error(actionErrorRuntimeError, err.str());
        return;

    }else if(start > evictweights->latest){
        err << "EvictWeights could not start in time"
            << " (now " << util::millis(start)
            << ", latest " << util::millis(evictweights->latest) << ")";
        error(actionErrorCouldNotStartInTime, err.str());
        return;
    }

    RuntimeModelDummy* rm = myManager->models->get(evictweights->model_id, evictweights->gpu_id);
    if (rm == nullptr) {
        error(evictWeightsUnknownModel, "EvictWeightsTask could not find model with specified id");
        return;
    }

    rm->lock();

    rm->version++;
    bool previous_weights = rm->weights;
    rm->weights = false;

    rm->unlock();

    if (!previous_weights) {
        error(evictWeightsNotInCache, "EvictWeightsTask not processed because no weights exist");
        return;
    }

    myManager->weights_caches[evictweights->gpu_id]->free(rm->weightspagescount);

    end = util::now();
    success();
}

void EvictWeightsDummy::success(){
    auto result = std::make_shared<workerapi::EvictWeightsResult>();

    result->id = evictweights->id;
    result->action_type = workerapi::evictWeightsAction;
    result->status = actionSuccess;

    //Set timestamps in the result
    result->begin = adjust_timestamp(start, -evictweights->clock_delta);
    result->end = adjust_timestamp(end, -evictweights->clock_delta);
    result->duration = result->end - result->begin;
    result->action_received = adjust_timestamp(evictweights->received, -evictweights->clock_delta);
    result->clock_delta = evictweights->clock_delta;
    
    myController->sendResult(result);
}

void EvictWeightsDummy::error(int status_code, std::string message){
    TaskError* error = new TaskError(status_code,message);
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->id = evictweights->id;
    result->action_type = workerapi::evictWeightsAction;
    result->status = error->status_code;
    result->message = error->message;
    result->action_received = adjust_timestamp(evictweights->received, -evictweights->clock_delta);
    result->clock_delta = evictweights->clock_delta;
    myController->sendResult(result);
    delete this;
}

void InferDummy::run(){
    start = util::now();

    std::stringstream err;
    if(start < infer->earliest){
        err << "Infer ran before it was eligible"
            << " (now " << util::millis(start)
            << ", earliest " << util::millis(infer->earliest) << ")";
        error(actionErrorRuntimeError, err.str());
        return;
    }else if(start > infer->latest){
        err << "Infer could not start in time"
            << " (now " << util::millis(start)
            << ", latest " << util::millis(infer->latest) << ")";
        error(actionErrorCouldNotStartInTime, err.str());
        return;
    }

    RuntimeModelDummy* rm = myManager->models->get(infer->model_id, infer->gpu_id);
    if (rm == nullptr) {
        error(copyInputUnknownModel, "CopyInputTask could not find model with specified id");
        return;
    }
    int padded_batch_size = rm->padded_batch_size(infer->batch_size);
    if (padded_batch_size == -1) {
        err << "CopyInputTask received unsupported batch size " << infer->batch_size;
        error(copyInputInvalidBatchSize, err.str());
        return;
    }
    if (infer->input_size == 0 && myManager->allow_zero_size_inputs) {
        // Used in testing; allow client to send zero-size inputs and generate worker-side
    }else if (rm->input_size(infer->batch_size) != infer->input_size) {
        // Normal behavior requires correctly sized inputs
        err << "CopyInputTask received incorrectly sized input"
            << " (expected " << rm->input_size(infer->batch_size) 
            << ", got " << infer->input_size
            << " (batch_size=" << infer->batch_size << ")";
        error(copyInputInvalidInput, err.str());
        return;
    }

    if (rm->weights == false) {
        error(execWeightsMissing, "ExecTask failed due to missing model weights");
    }

    rm->lock();
    version = rm->version;
    rm->unlock();
    end = start + rm->modelinfo->batch_size_exec_times_nanos[0]*padded_batch_size;
    element* action = new element();
    action->ready = end;
    action->callback = [this]() {this->process_completion();};
    myEngine->addToEnd(workerapi::inferAction,infer->gpu_id, action);
}

void InferDummy::process_completion(){
    RuntimeModelDummy* rm = myManager->models->get(infer->model_id, infer->gpu_id);
    bool version_unchanged = false;
    rm->lock();
    if (rm->version == version && rm->weights) {
        version_unchanged = true;
    }
    rm->unlock();
    if (version_unchanged) {
        success();
    } else {
        error(execConcurrentWeightsModification, "ExecTask failed due to weights version mismatch");
    }
}

void InferDummy::error(int status_code, std::string message){
    TaskError* error = new TaskError(status_code,message);
    auto result = std::make_shared<workerapi::ErrorResult>();
    result->id = infer->id;
    result->action_type = workerapi::inferAction;
    result->status = error->status_code;
    result->message = error->message;
    result->action_received = adjust_timestamp(infer->received, -infer->clock_delta);
    result->clock_delta = infer->clock_delta;
    myController->sendResult(result);
    delete this;
}

void InferDummy::success(){
    auto result = std::make_shared<workerapi::InferResult>();

    result->id = infer->id;
    result->action_type = workerapi::inferAction;
    result->status = actionSuccess;

    //Set timestamps in the result
    result->copy_input.begin = adjust_timestamp(start, -infer->clock_delta);
    result->exec.begin = adjust_timestamp(start, -infer->clock_delta);
    result->copy_output.begin = adjust_timestamp(end, -infer->clock_delta);
    result->copy_input.end = adjust_timestamp(start, -infer->clock_delta);
    result->exec.end = adjust_timestamp(end, -infer->clock_delta);
    result->copy_output.end = adjust_timestamp(end, -infer->clock_delta);
    result->copy_input.duration = result->copy_input.end - result->copy_input.begin;
    result->exec.duration = result->exec.end - result->exec.begin;
    result->copy_output.duration = result->copy_output.end - result->copy_output.begin;
    //QUESTION (uint64_t) (telemetry->async_duration * 1000000.0)?

    if (infer->input_size == 0) {
        result->output_size = 0;
    }else {
        RuntimeModelDummy* rm = myManager->models->get(infer->model_id, infer->gpu_id);
        result->output_size = rm->output_size(infer->batch_size);
    }
    result->output = (char*)nullptr;

    result->gpu_id = infer->gpu_id;
    result->gpu_clock_before = 1380;//Magic number for gpu_clock
    result->gpu_clock = 1380;
    
    result->clock_delta = infer->clock_delta;
    
    myController->sendResult(result);
    delete this;
}

}