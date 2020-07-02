#include "clockwork/worker_dummy.h"
#include <dmlc/logging.h>
#include <algorithm>
#include "clockwork/modeldef.h"

namespace clockwork {

void EngineDummy::enqueue(uint64_t end_at, std::function<void(void)> callback){
    if (end_at - util::now() <= 0) callback();
    else pending_actions.push(element{end_at, callback});
}

void EngineDummy::run() {
    while (alive.load()) {
        element next;
        while (pending_actions.try_pop(next)){
            if(util::now() >= next.ready) next.callback();
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

        LoadModelFromDiskDummy* loadmodel = new LoadModelFromDiskDummy(myManager,myEngine,action,myController);

        std::stringstream err;

        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at < action->earliest){
            err << "LoadModelFromDiskTask ran before it was eligible"
            << " (now " << util::millis(start_at)
            << ", earliest " << util::millis(action->earliest) << ")";
            std::string message = err.str();
            myEngine->enqueue(start_at, [loadmodel,message]() {
            loadmodel->error(actionErrorRuntimeError,message);});
        }
        else if(start_at > action->latest){
            err << "LoadModelFromDiskTask could not start in time"
            << " (now " << util::millis(start_at)
            << ", latest " << util::millis(action->latest) << ")";
            std::string message = err.str();
            myEngine->enqueue(start_at, [loadmodel,message]() {
            loadmodel->error(actionErrorCouldNotStartInTime, message);});
        }
        else{
            loadmodel->start = start_at;
            loadmodel->end = end_at;
            available_at = end_at;
            myEngine->enqueue(end_at, [loadmodel]() { loadmodel->run();});
        }
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::LoadWeights> action, uint64_t duration){
    if(alive){

        LoadWeightsDummy* loadweights = new LoadWeightsDummy(myManager,myEngine,action, myController);

        std::stringstream err;

        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at < action->earliest){
            err << "LoadWeights ran before it was eligible"
            << " (now " << util::millis(start_at)
            << ", earliest " << util::millis(action->earliest) << ")";
            std::string message = err.str();
            myEngine->enqueue(start_at, [loadweights,message]() {
            loadweights->error(actionErrorRuntimeError, message);});
        }
        else if(start_at > action->latest){
            err << "LoadWeights could not start in time"
            << " (now " << util::millis(start_at)
            << ", latest " << util::millis(action->latest) << ")";
            std::string message = err.str();
            myEngine->enqueue(start_at, [loadweights,message]() {
            loadweights->error(actionErrorCouldNotStartInTime, message);});
        }
        else{     
            RuntimeModelDummy* rm = myManager->models->get(action->model_id, action->gpu_id);
            if (rm == nullptr) {
                std::string message = "LoadWeightsTask could not find model";
                message += " with model ID " + std::to_string(action->model_id);
                message += " and GPU ID " + std::to_string(action->gpu_id);
                myEngine->enqueue(start_at, [loadweights,message]() {
                loadweights->error(loadWeightsUnknownModel, message);});
            }else{
                loadweights->start = start_at;
                loadweights->end = end_at;
                available_at = end_at;
                myEngine->enqueue(start_at, [loadweights]() { loadweights->run();});
                myEngine->enqueue(end_at, [loadweights]() { loadweights->process_completion();});
            }
        }
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::EvictWeights> action, uint64_t duration){
    if(alive){

        EvictWeightsDummy* evictweights = new EvictWeightsDummy(myManager,myEngine,action, myController);

        std::stringstream err;

        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at < action->earliest){
            err << "EvictWeights ran before it was eligible"
            << " (now " << util::millis(start_at)
            << ", earliest " << util::millis(action->earliest) << ")";
            std::string message = err.str();
            myEngine->enqueue(start_at, [evictweights,message]() {
            evictweights->error(actionErrorRuntimeError, message);});
        }
        else if(start_at > action->latest){
            err << "EvictWeights could not start in time"
            << " (now " << util::millis(start_at)
            << ", latest " << util::millis(action->latest) << ")";
            std::string message = err.str();
            myEngine->enqueue(start_at, [evictweights,message]() {
            evictweights->error(actionErrorCouldNotStartInTime, message);});
        }
        else{
            evictweights->start = start_at;
            evictweights->end = end_at;
            available_at = end_at;
            myEngine->enqueue(end_at, [evictweights]() {
            evictweights->run();});
        }
    }
};

void ExecutorDummy::new_action(std::shared_ptr<workerapi::Infer> action, uint64_t duration){
    if(alive){

        InferDummy* infer = new InferDummy(myManager,myEngine,action, myController, my_gpu_clock);

        std::stringstream err;

        uint64_t start_at = std::max(util::now(), available_at);
        uint64_t end_at = start_at + duration;
        if(start_at < action->earliest){
            err << "Infer ran before it was eligible"
            << " (now " << util::millis(start_at)
            << ", earliest " << util::millis(action->earliest) << ")";
            std::string message = err.str();
            myEngine->enqueue(start_at, [infer,message]() {
            infer->error(actionErrorRuntimeError, message);});
        }
        else if(start_at > action->latest){
            err << "Infer could not start in time"
            << " (now " << util::millis(start_at)
            << ", latest " << util::millis(action->latest) << ")";
            std::string message = err.str();
            myEngine->enqueue(start_at, [infer,message]() {
            infer->error(actionErrorCouldNotStartInTime, message);});
        }
        else{
            RuntimeModelDummy* rm = myManager->models->get(action->model_id, action->gpu_id);
            if (rm == nullptr) {
                myEngine->enqueue(start_at, [infer]() {
                infer->error(copyInputUnknownModel, "CopyInputTask could not find model with specified id");});
                return;
            }

            if (!rm->is_valid_batch_size(action->batch_size)) {
                std::stringstream err;
                err << "CopyInputTask received unsupported batch size " << action->batch_size;
                std::string message = err.str();
                myEngine->enqueue(start_at, [infer,message]() {
                infer->error(copyInputInvalidBatchSize, message);});
                return;
            }
            if (action->input_size == 0 && myManager->allow_zero_size_inputs) {
                // Used in testing; allow client to send zero-size inputs and generate worker-side
            } else if (rm->input_size(action->batch_size) != action->input_size) {
                // Normal behavior requires correctly sized inputs
                std::stringstream err;
                err << "CopyInputTask received incorrectly sized input"
                    << " (expected " << rm->input_size(action->batch_size) 
                    << ", got " << action->input_size
                    << " (batch_size=" << action->batch_size << ")";
                std::string message = err.str();
                myEngine->enqueue(start_at, [infer,message]() {
                infer->error(copyInputInvalidInput, message);});
                return;  //TODO WEI
            }
            infer->start = start_at;
            infer->end = end_at;
            available_at = end_at;
            myEngine->enqueue(start_at, [infer]() { infer->run();});
            myEngine->enqueue(end_at, [infer]() { infer->process_completion();});
        }
    }
};

void ClockworkRuntimeDummy::setController(workerapi::Controller* Controller){
    for (unsigned gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            
            gpu_executors[gpu_id]->setController(Controller);
            weights_executors[gpu_id]->setController(Controller);
            inputs_executors[gpu_id]->setController(Controller);
            outputs_executors[gpu_id]->setController(Controller);

        }
        load_model_executor->setController(Controller);
}

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

        runtime->load_model_executor->new_action(load_model,1000);
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

        runtime->weights_executors[load_weights->gpu_id]->new_action(load_weights,1000);      
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

        runtime->outputs_executors[evict_weights->gpu_id]->new_action(evict_weights,1000);
        
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

        runtime->gpu_executors[infer->gpu_id]->new_action(infer,1000);
    } else {
        invalidAction(action);
    }
}

void ClockworkDummyWorker::clearCache(std::shared_ptr<workerapi::Action> action) {
    auto clear_cache = std::static_pointer_cast<workerapi::ClearCache>(action);
    if (clear_cache != nullptr) {
        for (unsigned i = 0; i < runtime->num_gpus; i++) {
            runtime->manager->weights_caches[i]->clear();
        }// TODO WEI
        auto result = std::make_shared<workerapi::ClearCacheResult>();
        result->id = action->id;
        result->action_type = workerapi::clearCacheAction;
        result->status = actionSuccess; // TODO What about error handling?
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
        runtime->manager->get_worker_memory_info(result->worker);// TODO WEI
        result->status = actionSuccess; // TODO What about error handling?
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
    int page_size = myManager->weights_caches[0]->page_size;// TODO WEI
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

    size_t inputs_size = 0;
    size_t outputs_size = 0;
    unsigned weights_pages_count;
    std::vector<unsigned> supported_batch_sizes;
    std::vector<uint64_t> batch_size_exec_times_nanos;
    uint64_t weights_load_time_nanos;
    try{
        // Load data for batch sizes and extract performance profile
        std::vector<ModelDataDummy> modeldata = loadModelDataDummy(loadmodel->model_path);//TODO WEI
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
    
        for (auto &input : spec->inputs) {
            inputs_size += input.size;
        }

        for (auto &output : spec->outputs) {
            outputs_size += output.size;
        }

        //Add model to modelstore
        for (auto &gpu_id : gpu_ids) {

            for (unsigned i = 0; i < loadmodel->no_of_copies; i++) {

                workerapi::ModelInfo* modelInfo;
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

    }catch (dmlc::Error &err) {
        error(actionErrorInvalidModelID, err.what());
        return;
    }catch(NoMeasureFile &err){
        error(err.status_code, err.message);
        return;
    }
    success(inputs_size,outputs_size,weights_pages_count,weights_load_time_nanos,supported_batch_sizes,batch_size_exec_times_nanos);

}



void LoadWeightsDummy::run(){
    RuntimeModelDummy* rm = myManager->models->get(loadweights->model_id, loadweights->gpu_id);
    rm->lock();
    if (!rm->weights) {
        //TODO alloc weight;
        alloc_success = myManager->weights_caches[loadweights->gpu_id]->alloc(rm->weightspagescount);
    }
    version = ++rm->version;
    rm->weights = true;
    rm->unlock();
}

void LoadWeightsDummy::process_completion(){
    RuntimeModelDummy* rm = myManager->models->get(loadweights->model_id, loadweights->gpu_id);
    bool version_unchanged = false;
    rm->lock();
    if (rm->version == version && rm->weights) {
        version_unchanged = true;
    }
    rm->unlock();
    if (version_unchanged) {
        if(alloc_success){
            success();
        }else{
            error(loadWeightsInsufficientCache, "LoadWeightsTask failed to allocate pages from cache");
            rm->lock();
            rm->weights = false;
            rm->unlock();
        }
    } else {
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
    result->duration = result->end - result->begin;//QUESTION (uint64_t) (telemetry->async_duration * 1000000.0)?
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
    result->duration = result->end - result->begin;//QUESTION (uint64_t) (telemetry->async_duration * 1000000.0)?
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
    gpu_clock_before = gpu_clock->get(infer->gpu_id);
    RuntimeModelDummy* rm = myManager->models->get(infer->model_id, infer->gpu_id);
    rm->lock();
    version = rm->version;
    rm->unlock();
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
    result->copy_output.begin = adjust_timestamp(start, -infer->clock_delta);
    result->copy_input.end = adjust_timestamp(start, -infer->clock_delta);
    result->exec.end = adjust_timestamp(end, -infer->clock_delta);
    result->copy_output.end = adjust_timestamp(start, -infer->clock_delta);
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
    result->gpu_clock_before = gpu_clock_before;
    result->gpu_clock = gpu_clock->get(result->gpu_id);
    
    result->clock_delta = infer->clock_delta;
    
    myController->sendResult(result);
    delete this;
}

}
