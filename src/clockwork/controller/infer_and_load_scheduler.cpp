#include "clockwork/controller/infer_and_load_scheduler.h"

namespace clockwork {
namespace scheduler {
namespace infer3 {

uint64_t action_id_seed = 0;

Scheduler::WorkTracker::WorkTracker(int num_gpus, int num_models) {
    gpus.resize(num_gpus);
    for (unsigned i = 0; i < num_gpus; i++) {
        gpus[i].id = i;
        gpus[i].models.resize(num_models, false);
    }

    models.resize(num_models);
    ordered.reserve(num_models);
    for (unsigned i = 0; i < num_models; i++) {
        models[i].id = i;
        models[i].gpus.resize(num_gpus, false);
        models[i].loading.resize(num_gpus, false);

        ordered.push_back(&models[i]);
    }
}

Scheduler::WorkTracker::Demand Scheduler::WorkTracker::addRequest(int model_id, int64_t size, uint64_t slo) {
    size = (size * capacity) / slo;

    Scheduler::WorkTracker::Demand demand;
    demand.model = model_id;
    demand.gpu = -1;
    demand.work = 0;
    demand.deficit = size;

    for (auto &gpu : gpus) {
        if (gpu.models[model_id] && gpu.work < capacity) {
            demand.gpu = gpu.id;
            auto amount = std::min(size, capacity - gpu.work);
            demand.deficit -= amount;
            demand.work += amount;
            gpu.work += demand.work;
            break;
        }
    }

    models[model_id].deficit += demand.deficit;
    models[model_id].work += demand.work;
    models[model_id].count++;

    return demand;
}

void Scheduler::WorkTracker::requestCompleted(Demand &demand) {
    if (demand.gpu != -1) {
        gpus[demand.gpu].work -= demand.work;
    }
    auto &model = models[demand.model];
    int64_t surplus = std::min(demand.deficit, model.surplus);
    model.surplus -= surplus;
    model.deficit -= demand.deficit - surplus;
    model.work -= demand.work;
    model.seqno = seqno_seed++;
    model.count--;
}

int Scheduler::WorkTracker::loadModel(int gpu_id, bool requires_eviction) {
    std::sort(ordered.begin(), ordered.end(), sort_by_deficit);

    auto &gpu = gpus[gpu_id];

    unsigned seen = 0;
    for (auto &model : ordered) {
        if (requires_eviction && seen == gpu.model_count) break;

        if (!model->gpus[gpu_id] && !model->loading[gpu_id]) {
            // Mark this model as loaded on this GPU
            model->gpus[gpu_id] = true;
            model->loading[gpu_id] = true;
            gpu.models[model->id] = true;
            gpu.model_count++;

            // Estimate how much work this GPU will take
            int64_t delta = model->deficit;
            if (model->gpu_count > 0) {
                delta = std::min(delta, (model->surplus + model->work) / model->gpu_count);
            }

            // Adjust the model counters
            model->deficit -= delta;
            model->surplus += delta;
            model->gpu_count++;

            return model->id;
        }

        if (model->gpus[gpu_id]) {
            seen++;
        }
    }

    return -1; // all models loaded on all gpus
}

void Scheduler::WorkTracker::loadModelComplete(int gpu_id, int model_id, bool success) {
    auto &model = models[model_id];
    auto &gpu = gpus[gpu_id];

    model.loading[gpu_id] = false;

    model.gpus[gpu_id] = success;
    gpu.models[model_id] = success;

    if (!success) {
        model.gpu_count--;
        model.deficit += model.surplus;
        model.surplus = 0;
    }

}

int Scheduler::WorkTracker::evictModel(int gpu_id) {
    std::sort(ordered.begin(), ordered.end(), sort_by_deficit);

    auto &gpu = gpus[gpu_id];

    for (unsigned i = ordered.size()-1; i >= 0; i--) {
        Model* model = ordered[i];
        if (model->gpus[gpu_id] && !model->loading[gpu_id]) {
            // Unmark the model as loaded on this GPU
            model->gpus[gpu_id] = false;
            gpu.models[model->id] = false;
            model->gpu_count--;
            gpu.model_count--;

            return model->id;
        }
    }

    return -1; // all models are unloaded on all gpus
}

Scheduler::Scheduler(std::string actions_filename) 
    : actions_filename(actions_filename) {
}

Scheduler::RequestImpl::RequestImpl(clientapi::InferenceRequest request,
    std::function<void(clientapi::InferenceResponse&)> callback) : 
        request(request), 
        callback(callback) {
    // Set the response header fields now
    response.header.user_request_id = request.header.user_request_id;
    response.header.message = "";
    response.model_id = request.model_id;
    response.batch_size = request.batch_size;
    response.output = nullptr;
    response.output_size = 0;
    response.deadline = request.arrival + Scheduler::slo;

    this->deadline = response.deadline - Scheduler::buffer;
}

Scheduler::RequestImpl::~RequestImpl() {
    delete static_cast<char*>(request.input);
}

void Scheduler::RequestImpl::set_result(char* output, size_t output_size) {
    response.header.status = clockworkSuccess;
    response.output = output;
    response.output_size = output_size;
}

void Scheduler::RequestImpl::set_error(int status, std::string message) {
    response.header.status = status;
    response.header.message = message;
}

bool Scheduler::RequestImpl::complete(uint64_t now) {
    if (has_completed) return false; // already timed out
    has_completed = true;

    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    // Set the departure time (controller.cpp can also do this, 
    // but we want to report departure time back to the action to determine goodput)
    this->departure = now;
    response.departure = now;

    callback(response);

    model->scheduler->tracker->requestCompleted(demand);

    return response.header.status == clockworkSuccess && departure <= response.deadline;
}

void Scheduler::RequestImpl::timeout() {
    if (has_completed) return; // already timed out
    has_completed = true;

    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    response.header.status = clockworkTimeout;
    response.departure = util::now();

    callback(response);

    model->scheduler->tracker->requestCompleted(demand);
}

void Scheduler::RequestImpl::finalize() {
    if (!has_completed) timeout();

    // model->scheduler->tracker->requestCompleted(demand);
}

unsigned Scheduler::Model::batch_lookup(unsigned num_requests) {
    return num_requests > max_batch_size ? max_batch_size : batch_lookup_[num_requests];
}

Scheduler::Model::Model(BatchedModelState &state)
    : id(state.id), num_weights_pages(state.num_weights_pages) {
    for (auto &batch_size : state.supported_batch_sizes) {

        uint64_t estimate = 0; // Default 0.1ms estimate

        // Lookup real estimate if it was provided
        auto it = state.exec_duration.find(batch_size);
        if (it != state.exec_duration.end()) {
            estimate = it->second;
        }

        // Limit the batch sizes we use
        if (batch_size == 1 || 
            (estimate > 0 && estimate < Scheduler::max_allowable_exec_time)) {
            estimates[batch_size] = estimate * Scheduler::default_clock;
            estimators[batch_size] = new util::SlidingWindow(Scheduler::estimate_window_size);
            supported_batch_sizes.push_back(batch_size);
        } else {
            std::cout << "Excluding b" << batch_size << " with estimate " << estimate << "model=" << state.model_path << std::endl;
        }
    }

    weights_estimator = new util::SlidingWindow(Scheduler::estimate_window_size);
    weights_estimate = state.weights_transfer_duration;

    batch_lookup_ = util::make_batch_lookup(supported_batch_sizes);
    max_batch_size = batch_lookup_.size() - 1;
}

void Scheduler::Model::enqueue(Request request) {
    // Enqueue the actual request to the model
    request->id = request_id_seed++;
    queue.push_back(request);

    // Get batch size info
    std::vector<uint64_t> batch_size_estimates(supported_batch_sizes.size());
    for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
        batch_size_estimates[i] = estimate(supported_batch_sizes[i]);
    }

    // Enqueue strategies to all loaded models
    for (auto &instance : instances) {
        if (instance->loaded) {
            for (unsigned i = 0; i < batch_size_estimates.size(); i++) {
                auto strategy = new Scheduler::InferStrategy();
                strategy->priority = request->deadline - batch_size_estimates[i];
                strategy->deadline = request->deadline;
                strategy->request_id = request->id;
                strategy->batch_size = supported_batch_sizes[i];
                strategy->instance = instance;
                strategy->version = instance->version;

                instance->gpu->queue.push(strategy);
            }
        }
    }

    // Add the request to the load tracker
    unsigned i = supported_batch_sizes.size()-1;
    uint64_t size_for_tracker = batch_size_estimates[1];// / supported_batch_sizes[1];

    uint64_t slo = (request->deadline - util::now()) - estimate_weights() - batch_size_estimates[i]; // for execution
    request->demand = scheduler->tracker->addRequest(id, size_for_tracker, slo);
}

void Scheduler::Model::enqueue(ModelInstance* instance) {
    std::vector<uint64_t> batch_size_estimates(supported_batch_sizes.size());
    for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
        batch_size_estimates[i] = estimate(supported_batch_sizes[i]);
    }
    
    for (auto &request : queue) {
        for (unsigned i = 0; i < batch_size_estimates.size(); i++) {
            auto strategy = new Scheduler::InferStrategy();
            strategy->priority = request->deadline - batch_size_estimates[i];
            strategy->deadline = request->deadline;
            strategy->request_id = request->id;
            strategy->batch_size = supported_batch_sizes[i];
            strategy->instance = instance;
            strategy->version = instance->version;

            instance->gpu->queue.push(strategy);
        }
    }
}

void Scheduler::Model::check_timeouts(uint64_t free_at) {
    while (!queue.empty()) {
        Request request = queue.front();
        if (request->deadline >= free_at && !request->has_completed) break;

        queue.pop_front();
        request->timeout();
    }
}

Scheduler::InferAction* Scheduler::Model::try_dequeue(
        GPU* gpu, 
        uint64_t free_at, 
        Scheduler::InferStrategy* strategy)
{
    // Drop any timed out requests
    check_timeouts(free_at);

    // No requests queued
    if (queue.empty()) return nullptr;

    // The model instance was updated since this strategy was created
    if (strategy->instance->version != strategy->version) return nullptr;

    // The request that generated this strategy has already completed (e.g. as part of a previous batch)
    if (queue.front()->id > strategy->request_id) return nullptr;

    // See if this strategy has enough requests to fill its batch
    //   ie. that (batch_size-1) new requests arrived after this request arrived
    // Note that this is not simply queue.size()
    if (request_id_seed - strategy->request_id < strategy->batch_size) return nullptr;

    // See if a greater batch size might be achievable by a subsequent request
    if (strategy->batch_size < batch_lookup(request_id_seed - strategy->request_id - 1)) return nullptr;

    // See if the strategy can actually execute given the current GPU clock
    unsigned gpu_clock = gpu->exec.clock();
    uint64_t exec_time = estimate(strategy->batch_size, gpu_clock);
    uint64_t completion_time = free_at + exec_time;
    if (completion_time > strategy->deadline) return nullptr;

    // We are good to go.  Drop any requests that came before this strategy
    while (queue.front()->id != strategy->request_id) {
        auto request = queue.front();
        queue.pop_front();
        request->timeout();
    }

    CHECK(queue.size() >= strategy->batch_size) << "Scheduler logic error";

    InferAction* action = new InferAction(this);
    for (unsigned i = 0; i < strategy->batch_size; i++) {
        action->requests.push_back(queue.front());
        queue.pop_front();
    }
    action->set_expectations(free_at, exec_time, gpu_clock);
    action->batch();

    return action;
}


const float Scheduler::estimate_percentile = 0.99;

void Scheduler::Model::add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock) {
    auto it = estimators.find(batch_size);
    CHECK(it != estimators.end()) << "Unsupported batch size " << batch_size;
    auto estimator = it->second;
    estimator->insert(duration * gpu_clock);

    estimates[batch_size] = estimator->get_percentile(Scheduler::estimate_percentile);
}

void Scheduler::Model::add_weights_measurement(uint64_t duration) {
    weights_estimator->insert(duration);
    weights_estimate = weights_estimator->get_percentile(Scheduler::estimate_percentile);
}

uint64_t Scheduler::Model::estimate(unsigned batch_size) {
    return Scheduler::Model::estimate(batch_size, Scheduler::default_clock);
}

uint64_t Scheduler::Model::estimate(unsigned batch_size, int clock) {
    unsigned effective_batch_size = batch_lookup(batch_size);
    return estimates[effective_batch_size] / clock;
}

uint64_t Scheduler::Model::estimate_weights() {
    return weights_estimate;
}

Scheduler::InferAction::InferAction(Model* model) : model(model) {
    action->id = action_id_seed++;
    action->model_id = model->id;
}

Scheduler::InferAction::~InferAction() {
    if (result != nullptr) {
        delete result->output;
        delete action->input;
    }
}

void Scheduler::InferAction::batch() {
    action->batch_size = requests.size();
    action->input_size = 0;
    for (auto &r : requests) {
        action->input_size += r->request.input_size;
    }
    action->input = new char[action->input_size];
    size_t offset = 0;
    for (auto &r : requests) {
        std::memcpy(action->input + offset, r->request.input, r->request.input_size);
        offset += r->request.input_size;
    }
}

void Scheduler::InferAction::unbatch() {
    size_t single_output_size = result->output_size / requests.size();
    size_t offset = 0;
    for (unsigned i = 0; i < requests.size(); i++) {
        char* output = new char[single_output_size];
        std::memcpy(output, result->output + offset, single_output_size);
        offset += single_output_size;

        requests[i]->set_result(output, single_output_size);
    }
}

void Scheduler::InferAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    for (auto &request : requests) {
        request->set_error(clockworkError, error->message);
    }
}

void Scheduler::InferAction::set_result(std::shared_ptr<workerapi::InferResult> &result) {
    this->result = result;
    this->unbatch();
}

float Scheduler::InferAction::complete(uint64_t now) {
    float successful_requests = 0;
    float total_requests = 0;
    for (auto &request : requests) {
        if (request->complete(now)) {
            successful_requests += 1;
        }
        total_requests += 1;
    }
    return successful_requests / total_requests;
}

void Scheduler::InferAction::set_expectations(uint64_t exec_start, uint64_t duration, int clock) {
    action->expected_duration = duration;
    action->expected_exec_complete = exec_start + duration;
    action->expected_gpu_clock = clock;
    action->earliest = exec_start;
    action->latest = action->earliest + Scheduler::latest_delta;
    // action->earliest = util::now() - Scheduler::schedule_ahead;
    // action->latest = action->expected_exec_complete + Scheduler::latest_delta;
}

        // ModelInstance* instance;
        // unsigned version;
        // ControllerActionTelemetry telemetry;
        // std::shared_ptr<workerapi::LoadWeights> action = std::make_shared<workerapi::LoadWeights>();
        // std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        // std::shared_ptr<workerapi::LoadWeightsResult> result = nullptr;

Scheduler::LoadWeightsAction::LoadWeightsAction(ModelInstance* instance) : instance(instance) {
    action->id = action_id_seed++;
    action->model_id = instance->model->id;
}

void Scheduler::LoadWeightsAction::set_expectations(uint64_t exec_start, uint64_t duration) {
    action->expected_duration = duration;
    action->expected_exec_complete = exec_start + duration;
    action->earliest = util::now() - Scheduler::schedule_ahead;
    action->latest = action->expected_exec_complete + Scheduler::latest_delta;
}

void Scheduler::LoadWeightsAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    if (version = instance->version) {
        instance->version++;
        instance->loaded = false;
    }
}

void Scheduler::LoadWeightsAction::set_result(std::shared_ptr<workerapi::LoadWeightsResult> &result) {
    this->result = result;
}

Scheduler::EvictWeightsAction::EvictWeightsAction(ModelInstance* instance) : instance(instance) {
    action->id = action_id_seed++;
    action->model_id = instance->model->id;
}

void Scheduler::EvictWeightsAction::set_expectations() {
    action->earliest = 0;
    action->latest = UINT64_MAX;
}

void Scheduler::EvictWeightsAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    CHECK(false) << "Error in EvictWeightsAction" << error->str();
}

void Scheduler::EvictWeightsAction::set_result(std::shared_ptr<workerapi::EvictWeightsResult> &result) {
    this->result = result;
}

Scheduler::GPU::GPU() : exec(Scheduler::default_clock), loadweights(Scheduler::default_clock) {
}

void Scheduler::GPU::send_action(InferAction* action) {
    auto &infer = action->action;
    infer->gpu_id = gpu_id;
    infer->worker_id = worker_id;

    // Send the action first
    worker->sendAction(infer);

    // Update GPU state
    exec.add(infer->id, infer->expected_duration);

    // Save the callback
    scheduler->outstanding_actions[infer->id] = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->infer_result(action, result);
    };

    // Record the telemetry
    action->telemetry.set(infer);

    if (print_debug) std::cout << ("Worker <--  " + infer->str() + "\n");
}

void Scheduler::GPU::send_action(LoadWeightsAction* action) {
    auto &load = action->action;
    load->gpu_id = gpu_id;
    load->worker_id = worker_id;

    // Send the action first
    worker->sendAction(load);

    // Update PCI state
    loadweights.add(load->id, load->expected_duration);

    // Save the callback
    scheduler->outstanding_actions[load->id] = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->load_result(action, result);
    };

    // Record the telemetry
    action->telemetry.set(load);

    if (print_debug || print_loads) std::cout << ("Worker <--  " + load->str() + "\n");
}

void Scheduler::GPU::send_action(EvictWeightsAction* action) {
    auto &evict = action->action;
    evict->gpu_id = gpu_id;
    evict->worker_id = worker_id;

    // Send the action first
    worker->sendAction(evict);

    // Save the callback
    scheduler->outstanding_actions[evict->id] = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->evict_result(action, result);
    };

    // Record the telemetry
    action->telemetry.set(evict);

    if (print_debug || print_loads) std::cout << ("Worker <--  " + evict->str() + "\n");    
}

void Scheduler::GPU::evict_pages(unsigned required_pages) {
    while (free_pages < required_pages) {
        int model_id = scheduler->tracker->evictModel(id);

        if (model_id == -1) break;

        ModelInstance* instance = instances[model_id];
        instance->version++;
        instance->loaded = false;

        EvictWeightsAction* evict = new EvictWeightsAction(instance);
        evict->set_expectations();

        send_action(evict);
        
        free_pages += scheduler->models[model_id]->num_weights_pages;
        eviction_required = true; // GPU reached capacity; evictions required in future
    }
}

void Scheduler::GPU::check_pending() {
    uint64_t available = exec.available();

    // First, see if any model instances are at the point where their actions can be scheduled
    while (loading.size() > 0) {
        auto load = loading.front();

        if (load.version == load.instance->version && load.instance->loaded) {
            if (load.available_at > available) break;

            // Ready to enqueue infer strategies for this model
            load.instance->model->enqueue(load.instance);
        }
        loading.pop();
    }

    // Now check for any infer actions
    uint64_t schedule_until = util::now() + schedule_ahead;
    while ((available = exec.available()) < schedule_until && queue.size() > 0) {
        auto strategy = queue.top();
        queue.pop();

        InferAction* action = strategy->instance->model->try_dequeue(this, available, strategy);
        if (action != nullptr) {
            send_action(action);
        }
        delete strategy;
    }

    // Now check for any load actions
    while ((available = loadweights.available()) < schedule_until) {
        int model_id = scheduler->tracker->loadModel(id, eviction_required);
        if (model_id == -1) break;

        ModelInstance* instance = instances[model_id];
        CHECK(instance->loaded == false) << "Tracker asked to load model that is already loaded";

        unsigned size = scheduler->models[model_id]->num_weights_pages;
        evict_pages(size);

        if (free_pages < size) {
            scheduler->tracker->loadModelComplete(id, model_id, false);
            break;
        }

        instance->version++;
        instance->loaded = true;
        free_pages -= size;

        uint64_t expected_duration = instance->model->estimate_weights();

        LoadWeightsAction* action = new LoadWeightsAction(instance);
        action->version = instance->version;
        action->set_expectations(available, expected_duration);

        // uint64_t exec_may_begin_at = std::max(available, util::now() + latest_delta) + expected_duration;
        // loading.push({instance, instance->version, exec_may_begin_at});

        send_action(action);
    }
}

void Scheduler::GPU::infer_error(InferAction* action, std::shared_ptr<workerapi::ErrorResult> &error) {
    // std::cout << ("Worker  --> " + error->str() + "\n");

    action->telemetry.set(error);
    
    // Update GPU state tracking
    exec.error(error->id, util::now());

    // // Update model load tracking
    // for (auto &request : action->requests) {
    //     scheduler->tracker->requestCompleted(request->demand);
    // }

    action->set_error(error);
    CHECK(action->complete(util::now()) == 0) << "ErrorResult should not result in successful requests";

    action->telemetry.goodput = 0;
    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::infer_success(InferAction* action, std::shared_ptr<workerapi::InferResult> &result) {
    action->telemetry.set(result);

    // Update GPU state tracking
    exec.success(result->id, result->exec.end);
    exec.update_clock(result->gpu_clock);

    // // Update model load tracking
    // for (auto &request : action->requests) {
    //     scheduler->tracker->requestCompleted(request->demand);
    // }

    // Update model execution tracking
    action->model->add_measurement(
        action->action->batch_size, 
        result->exec.duration, 
        (result->gpu_clock + result->gpu_clock_before) / 2
    );
    // action->model->add_measurement(action->action->batch_size, result->exec.duration, action->action->expected_gpu_clock);

    action->set_result(result);
    action->telemetry.goodput = action->complete(util::now());

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::infer_result(InferAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        infer_error(action, error);

    } else if (auto infer = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
        infer_success(action, infer);

    } else {
        CHECK(false) << "Unexpected response to Infer action" << result->str();

    }
}

void Scheduler::GPU::load_error(LoadWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &error){
    action->telemetry.set(error);
    action->telemetry.goodput = 0;
    action->set_error(error);

    // Track model status
    scheduler->tracker->loadModelComplete(action->instance->gpu->id, action->instance->model->id, false);
    free_pages += action->instance->model->num_weights_pages;

    // Update PCI state tracking
    loadweights.error(error->id, util::now());

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::load_success(LoadWeightsAction* action, std::shared_ptr<workerapi::LoadWeightsResult> &result) {
    action->telemetry.set(result);
    action->set_result(result);

    // Track model status
    scheduler->tracker->loadModelComplete(action->instance->gpu->id, action->instance->model->id, true);

    // Update PCI state tracking
    loadweights.success(result->id, result->end);

    // Update PCI tracking
    action->instance->model->add_weights_measurement(result->duration);

    // // Enable new requests
    loading.push({action->instance, action->instance->version, 0});

    // TODO: change this?
    action->telemetry.goodput = 1.0;

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::load_result(LoadWeightsAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (!print_debug && print_loads) std::cout << ("Worker  --> " + result->str() + "\n");

    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        load_error(action, error);

    } else if (auto load = std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(result)) {
        load_success(action, load);

    } else {
        CHECK(false) << "Unexpected response to LoadWeights action" << result->str();

    }    
}

void Scheduler::GPU::evict_error(EvictWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &error){
    action->telemetry.set(error);

    action->set_error(error);

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::evict_success(EvictWeightsAction* action, std::shared_ptr<workerapi::EvictWeightsResult> &result) {
    action->telemetry.set(result);

    action->set_result(result);

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::evict_result(EvictWeightsAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (!print_debug && print_loads) std::cout << ("Worker  --> " + result->str() + "\n");

    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        evict_error(action, error);

    } else if (auto evict = std::dynamic_pointer_cast<workerapi::EvictWeightsResult>(result)) {
        evict_success(action, evict);

    } else {
        CHECK(false) << "Unexpected response to LoadWeights action" << result->str();

    }
}

void Scheduler::validate_clockwork_state(ClockworkState &state) {
    unsigned cache_size = state.workers[0].gpus[0].weights_cache_total_pages;
    for (auto &worker : state.workers) {
        for (auto &gpu : worker.gpus) {
            CHECK(gpu.weights_cache_total_pages == cache_size) 
                << "Expect same cache size on all GPUs";
        }
    }

    for (auto &p : state.workers[0].models) {
        unsigned model_id = p.first;
        for (auto &worker : state.workers) {
            CHECK(worker.models.find(model_id) != worker.models.end()) 
                << "Inconsistent models across workers";
        }
    }
}

void Scheduler::print_status() {
    unsigned model_pages = 0;
    for (auto &model : models) {
        model_pages += model->num_weights_pages;
    }

    unsigned gpu_pages = 0;
    for (auto &gpu : gpus) {
        gpu_pages += gpu->free_pages;
    }


    std::cout << "Total GPU capacity " << gpu_pages << " pages (" << gpu_pages/gpus.size() << " per GPU)." << std::endl
              << "Total model pages " << model_pages << " (" << (100*model_pages / gpu_pages) << "% oversubscription." << std::endl;
}

void Scheduler::initialize_models(ClockworkState &state) {
    models.resize(state.workers[0].models.size(), nullptr);

    for (auto &p : state.workers[0].models) {
        auto &model = p.second;

        if (model.id >= models.size()) {
            models.resize(model.id+1, nullptr);
        }

        models[model.id] = new Model(model);
        models[model.id]->scheduler = this;
    }

    std::cout << "Created " << models.size() << " models" << std::endl;
}

void Scheduler::initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                ClockworkState &state) 
{
    unsigned total_pages = 0;
    for (WorkerState &worker : state.workers) {
        for (GPUState &gpustate : worker.gpus) {
            GPU* gpu = new GPU();
            gpu->id = gpus.size();
            gpu->scheduler = this;
            gpu->worker = workers[worker.id];
            gpu->worker_id = worker.id;
            gpu->gpu_id = gpustate.id;
            gpu->free_pages = gpustate.weights_cache_total_pages;
            gpus.push_back(gpu);

            total_pages += gpu->free_pages;
        }
    }
    std::cout << "Created " << gpus.size() << " GPUs on " << state.workers.size() << " Workers" << std::endl;
}

void Scheduler::initialize_model_instances() {
    for (auto &gpu : gpus) {
        gpu->instances.resize(models.size(), nullptr);
    }
    for (auto &model : models) {
        model->instances.resize(gpus.size(), nullptr);
    }

    for (unsigned i = 0; i < gpus.size(); i++) {
        GPU* gpu = gpus[i];
        for (unsigned j = 0; j < models.size(); j++) {
            Model* model = models[j];
            ModelInstance* instance = new ModelInstance(gpu, model);
            model->instances[i] = instance;
            gpu->instances[j] = instance;
        }
    }

    tracker = new WorkTracker2(gpus.size(), models.size());
}


// Called when model loading has completed
void Scheduler::start(std::vector<network::controller::WorkerConnection*> workers,
                    ClockworkState &state) 
{
    validate_clockwork_state(state);
    initialize_models(state);
    initialize_gpus(workers, state);
    initialize_model_instances();
    print_status();

    this->thread = std::thread(&Scheduler::run, this);
    threading::initHighPriorityThread(this->thread);
}

void Scheduler::handle_request(Request request) {
    // Enqueue the request to the model
    unsigned model_id = request->request.model_id;
    if (model_id > models.size() || models[model_id] == nullptr) {
        request->set_error(clockworkError, "Invalid model ID");
        CHECK(!request->complete(util::now())) << "Erroneous request should not be successful";
        return;
    }

    Model* model = models[model_id];
    request->model = model;
    model->enqueue(request);
    requests.push(request);
}

void Scheduler::handle_result(std::shared_ptr<workerapi::Result> result) {
    auto it = outstanding_actions.find(result->id);
    CHECK(it != outstanding_actions.end()) 
        << "Received result for non-existent action " << result->str();

    auto &callback = it->second;
    callback(result);
    outstanding_actions.erase(it);
}

void Scheduler::run() {
    // Create and start the printer thread
    printer = ControllerActionTelemetry::log_and_summarize(actions_filename, print_interval);

    // Start processing actions + results
    unsigned start = 0;
    while (true) {
        Request request;
        while (request_queue.try_pop(request)) {
            handle_request(request);
        }

        std::shared_ptr<workerapi::Result> result;
        while (result_queue.try_pop(result)) {
            handle_result(result);
        }

        for (unsigned i = 0; i < gpus.size(); i++) {
            gpus[(i + start) % gpus.size()]->check_pending();
        }
        start = (start + 1) % gpus.size();

        // Drop any timed out requests
        uint64_t now = util::now();
        while (!requests.empty()) {
            auto &request = requests.front();

            if (request->response.deadline < now) {
                request->finalize();
                requests.pop();
            } else {
                break;
            }
        }
    }
}

// The actual scheduler interface implementation, invoked by worker network thread
void Scheduler::resultFromWorker(std::shared_ptr<workerapi::Result> result)
{
    if (print_debug) std::cout << ("Worker  --> " + result->str() + "\n");

    result->result_received = util::now();
    result_queue.push(result);
}

// The actual scheduler interface implementation, invoked by client network thread
void Scheduler::clientInfer(clientapi::InferenceRequest &request, 
    std::function<void(clientapi::InferenceResponse&)> callback)
{
    if (print_debug) std::cout << ("Client  --> " + request.str() + "\n");

    request_queue.push(std::make_shared<RequestImpl>(request, callback));
}

}
}
}