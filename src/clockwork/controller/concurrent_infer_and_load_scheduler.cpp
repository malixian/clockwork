#include "clockwork/controller/concurrent_infer_and_load_scheduler.h"

namespace clockwork {
namespace scheduler {
namespace infer4 {

std::atomic_uint64_t action_id_seed = 0;


Scheduler::Scheduler(uint64_t default_slo, uint64_t latest_delta,
                     uint64_t schedule_ahead, 
                     bool generate_inputs, int max_gpus,
                     uint64_t max_allowable_exec_time, unsigned max_batch_size,
                     std::string actions_filename)
    : default_slo(default_slo),
      latest_delta(latest_delta),
      schedule_ahead(schedule_ahead),
      max_allowable_exec_time(max_allowable_exec_time),
      max_batch_size(max_batch_size),
      generate_inputs(generate_inputs),
      max_gpus(max_gpus),
      actions_filename(actions_filename),
      actions_in_use(ATOMIC_FLAG_INIT) {
    std::cout << "ConcurrentInferAndLoadScheduler using:" << std::endl;
    std::cout << "\t default_slo=" << default_slo << std::endl;
    std::cout << "\t latest_delta=" << latest_delta << std::endl;
    std::cout << "\t schedule_ahead=" << schedule_ahead << std::endl;
    std::cout << "\t max_allowable_exec_time=" << max_allowable_exec_time << std::endl;
    std::cout << "\t max_batch_size=" << max_batch_size << std::endl;
    std::cout << "\t generate_inputs=" << generate_inputs << std::endl;
    std::cout << "\t max_gpus=" << max_gpus << std::endl;
}

void Scheduler::WorkTracker2::attach(Model &model) {
    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.loading[i]) {
            // Loading on a GPU is neither loadable nor evictable
        } else if (model.gpus[i]) {
            gpus[i].cached.insert(model.priorities[i]);
        } else {
            gpus[i].not_cached.insert(model.priorities[i]);
        }
    }
}

void Scheduler::WorkTracker2::detach(Model &model) {
    for (unsigned i = 0; i < n_gpus; i++) {
        auto &gpu = gpus[i];
        auto &priority = model.priorities[i];
        if (model.loading[i]) {
            // Loading on a GPU is neither loadable nor evictable
        } else if (model.gpus[i]) {
            auto it = gpu.cached.find(priority);
            CHECK(it != gpu.cached.end()) << "Thought we were cached when we weren't";

            gpu.cached.erase(it);
        } else {
            auto it = gpu.not_cached.find(priority);
            CHECK(it != gpu.not_cached.end()) << "Thought we were not cached when we were";

            gpu.not_cached.erase(it);
        }
    }
}

void Scheduler::WorkTracker2::updatePriority(Model &model) {
    // Calculate each GPU's weight
    double total_weight = 0;
    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            total_weight += gpus[i].weight;
        }
    }

    // Load priority is calculated differently to evict priority
    // First, load priority.  Load priority is simply whether we can satisfy outstanding_loadweights
    int64_t load_priority = model.outstanding_loadweights;

    // Subtract served load
    if (total_weight > 0 && load_priority > 0) {
        for (unsigned i = 0; i < n_gpus; i++) {
            if (model.gpus[i]) continue; // Skip models we are loaded on

            int64_t required = model.outstanding_loadweights * (gpus[i].weight / total_weight);
            int64_t served = (capacity * required) / gpus[i].outstanding;
            load_priority -= served;
        }
    }

    bool is_empty = model.outstanding_loadweights == 0 && model.outstanding_exec == 0;

    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            model.priorities[i]->priority = model.last_used[i];
        } else {
            model.priorities[i]->priority = load_priority;
        }
        model.priorities[i]->is_empty = is_empty;
        model.priorities[i]->last_used = model.last_used[i];
    }
}

void Scheduler::WorkTracker2::clearWork(Model &model) {
    for (unsigned i = 0; i < n_gpus; i++) {
        gpus[i].outstanding -= model.allocations[i];
        model.allocations[i] = 0;
    }
}

void Scheduler::WorkTracker2::distributeWork(Model &model) {
    // Update all the counters
    model.outstanding_exec -= model.completed_exec;
    model.completed_exec = 0;
    int64_t loadweights_delta = std::max(model.completed_loadweights, model.timedout_loadweights);
    model.outstanding_loadweights -= loadweights_delta;
    model.completed_loadweights -= loadweights_delta;
    model.timedout_loadweights -= loadweights_delta;

    clearWork(model);

    if (model.gpu_count == 0) return;

    // For demand tracking we use exec

    double total_weight = 0;
    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            total_weight += gpus[i].weight;
        }
    }

    for (unsigned i = 0; i < n_gpus; i++) {
        if (model.gpus[i]) {
            auto allocation = model.outstanding_exec * (gpus[i].weight / total_weight);
            model.allocations[i] = allocation;
            gpus[i].outstanding += allocation;
            gpus[i].weight = capacity / ((double) gpus[i].outstanding);
        }
    }
}

void Scheduler::WorkTracker2::addGPU(Model &model, GPU &gpu) {
    detach(model);

    model.gpus[gpu.id] = true;
    model.loading[gpu.id] = true;
    gpu.models[model.id] = true;
    model.priorities[gpu.id]->preference = model.gpu_count++;
    model.last_used[gpu.id] = seqno_seed++;

    distributeWork(model);
    updatePriority(model);

    attach(model);
}

void Scheduler::WorkTracker2::addGPUcomplete(Model &model, GPU &gpu) {
    detach(model);

    model.loading[gpu.id] = false;
    model.last_used[gpu.id] = seqno_seed++;

    distributeWork(model);
    updatePriority(model);

    attach(model);
}

void Scheduler::WorkTracker2::removeGPU(Model &model, GPU &gpu) {
    detach(model); 

    model.gpus[gpu.id] = false;
    model.loading[gpu.id] = false;
    gpu.models[model.id] = false;
    model.gpu_count--;
    for (unsigned i = 0; i < n_gpus; i++) {
        auto pref = model.priorities[gpu.id]->preference;
        if (model.priorities[i]->preference > pref) {
            model.priorities[i]->preference--;
        }
    }


    distributeWork(model);
    updatePriority(model);

    attach(model);
}

void Scheduler::WorkTracker2::checkRequests() {
    uint64_t now = util::now();
    while (!requests.empty() && requests.top().time < now) {
        auto &request = requests.top();
        auto &model = models[request.model_id];
        model.timedout_loadweights += request.loadweights_size;

        detach(model);
        distributeWork(model);
        updatePriority(model);
        attach(model);

        requests.pop();
    }
}

Scheduler::WorkTracker2::WorkTracker2(int num_gpus, int num_models, uint64_t capacity) : 
n_models(num_models), n_gpus(num_gpus), capacity(capacity) {
    gpus.resize(num_gpus);
    for (unsigned i = 0; i < num_gpus; i++) {
        gpus[i].id = i;
        gpus[i].models.resize(num_models, false);
    }

    models.resize(num_models);
    for (unsigned i = 0; i < num_models; i++) {
        auto &model = models[i];
        model.id = i;
        model.gpus.resize(num_gpus, false);
        model.loading.resize(num_gpus, false);
        model.allocations.resize(num_gpus, 0);
        model.last_used.resize(num_gpus, 0);
        for (unsigned i = 0; i < num_gpus; i++) {
            model.last_used[i] = seqno_seed++;
        }

        model.priorities.resize(num_gpus);
        for (unsigned j = 0; j < num_gpus; j++) {
            auto priority = new ModelPriority(&model);
            priority->last_used = model.last_used[j];
            model.priorities[j] = priority;
        }

        attach(model);
    }            
}

Scheduler::WorkTracker2::Demand Scheduler::WorkTracker2::addRequest(
        int model_id, int64_t size, uint64_t start_exec_by, uint64_t start_loadweights_by) {
    // Complete any pending requests
    checkRequests();

    // Demand is used to track actual entry and exit
    Scheduler::WorkTracker2::Demand demand;
    demand.exec_size = (size * capacity) / start_exec_by;
    demand.loadweights_size = (size * capacity) / start_loadweights_by;
    demand.model_id = model_id;

    // Request is used to track eligibility for weights loading
    Scheduler::WorkTracker2::Request request;
    request.loadweights_size = demand.loadweights_size;
    request.model_id = model_id;
    request.time = util::now() + start_loadweights_by;
    requests.push(request);

    // Get the model
    Model& model = models[model_id];
    model.outstanding_exec += demand.exec_size;
    model.outstanding_loadweights += demand.loadweights_size;
    
    // Update the model's priorities
    detach(model);
    distributeWork(model);
    updatePriority(model);
    attach(model);

    return demand;
}

void Scheduler::WorkTracker2::requestExecuting(Demand &demand, int gpu_id) {
    auto &model = models[demand.model_id];
    model.completed_loadweights += demand.loadweights_size;
    demand.loadweights_size = 0;
    if (gpu_id >= 0) model.last_used[gpu_id] = seqno_seed++;

    detach(model);
    distributeWork(model);
    updatePriority(model);
    attach(model);
}

void Scheduler::WorkTracker2::requestCompleted(Demand &demand, int gpu_id) {
    auto &model = models[demand.model_id];
    model.completed_exec += demand.exec_size;
    demand.exec_size = 0;
    if (gpu_id >= 0) model.last_used[gpu_id] = seqno_seed++;

    detach(model);
    distributeWork(model);
    updatePriority(model);
    attach(model);
}

void Scheduler::WorkTracker2::requestCancelled(Demand &demand, int gpu_id) {
    auto &model = models[demand.model_id];
    model.completed_exec += demand.exec_size;
    model.completed_loadweights += demand.loadweights_size;
    demand.exec_size = 0;
    demand.loadweights_size = 0;
    if (gpu_id >= 0) model.last_used[gpu_id] = seqno_seed++;

    detach(model);
    distributeWork(model);
    updatePriority(model);
    attach(model);
}

int Scheduler::WorkTracker2::loadModel(int gpu_id, bool requires_eviction) {
    // Complete any pending requests
    checkRequests();

    auto &gpu = gpus[gpu_id];
    if (gpu.not_cached.size() == 0) return -1;

    auto &priority = *gpu.not_cached.begin();
    if (priority->is_empty) return -1;
    if (priority <= 0) return -1; // all demand satisfied

    Model &model = *(priority->model);
    addGPU(model, gpu);
    return model.id;
}

void Scheduler::WorkTracker2::loadModelComplete(int gpu_id, int model_id, bool success) {
    // Complete any pending requests
    checkRequests();

    if (success) {
        addGPUcomplete(models[model_id], gpus[gpu_id]);
    } else {
        removeGPU(models[model_id], gpus[gpu_id]);        
    }
}

int Scheduler::WorkTracker2::evictModel(int gpu_id) {
    // Complete any pending requests
    checkRequests();

    auto &gpu = gpus[gpu_id];
    if (gpu.cached.size() == 0) return -1;

    auto &priority = *gpu.cached.rbegin();
    Model &model = *(priority->model);
    removeGPU(model, gpus[gpu_id]);
    return model.id;
}

Scheduler::RequestImpl::RequestImpl(
    Scheduler* scheduler,
    clientapi::InferenceRequest request,
    std::function<void(clientapi::InferenceResponse&)> callback) : 
        scheduler(scheduler),
        request(request), 
        callback(callback),
        locked(false) {
    // Set the response header fields now
    response.header.user_request_id = request.header.user_request_id;
    response.header.message = "";
    response.header.status = 0;
    response.model_id = request.model_id;
    response.batch_size = request.batch_size;
    response.output = nullptr;
    response.output_size = 0;
}

Scheduler::RequestImpl::~RequestImpl() {
    delete static_cast<char*>(request.input);
}

void Scheduler::RequestImpl::lock() {
    locked = true;
}

void Scheduler::RequestImpl::set_model(Model* model) {
    this->model = model;
    response.arrival_count = model->copies_loaded;
}

void Scheduler::RequestImpl::set_slo(uint64_t default_slo) {
    slo = default_slo;
    if (request.slo_factor > 0) {
        slo = model->b1_exec * request.slo_factor;
    }
    response.deadline = request.arrival + slo;

    exec_slo = std::min(slo, slo - Scheduler::buffer);
    exec_slo = std::max(exec_slo, scheduler->schedule_ahead + Scheduler::buffer);
    deadline = request.arrival + exec_slo;

    weights_slo = std::min(max_loadweights_slo, std::min(slo, slo - (model->estimate_weights() + model->b1_exec + Scheduler::buffer + scheduler->schedule_ahead)));
    weights_slo = std::max(weights_slo, scheduler->schedule_ahead + Scheduler::buffer);
}

void Scheduler::RequestImpl::set_result(char* output, size_t output_size) {
    response.header.status = clockworkSuccess;
    response.output = output;
    response.output_size = output_size;
    response.departure_count = model->copies_loaded;
}

void Scheduler::RequestImpl::set_error(int status, std::string message) {
    response.header.status = status;
    response.header.message = message;
}

bool Scheduler::RequestImpl::complete(uint64_t now, int gpu_id) {
    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    // Set the departure time (controller.cpp can also do this, 
    // but we want to report departure time back to the action to determine goodput)
    response.departure = now;

    callback(response);

    {
        tbb::queuing_mutex::scoped_lock lock(model->scheduler->tracker->mutex);
        model->scheduler->tracker->requestCompleted(demand, gpu_id);
    }

    return response.header.status == clockworkSuccess && response.departure <= response.deadline;
}

void Scheduler::RequestImpl::timeout() {
    if (locked) return;

    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    if (response.header.status == 0)
        response.header.status = clockworkTimeout;
    response.departure = util::now();
    response.departure_count = model->copies_loaded;

    callback(response);

    {
        tbb::queuing_mutex::scoped_lock lock(model->scheduler->tracker->mutex);
        model->scheduler->tracker->requestCancelled(demand, -1);
    }
}

void Scheduler::RequestImpl::finalize() {
    timeout();
}

unsigned Scheduler::Model::batch_lookup(unsigned num_requests) {
    return num_requests > max_batch_size ? max_batch_size : batch_lookup_[num_requests];
}

Scheduler::Model::Model(Scheduler* scheduler, BatchedModelState &state)
    : scheduler(scheduler),
      id(state.id), 
      num_weights_pages(state.num_weights_pages),
      weights_in_use(ATOMIC_FLAG_INIT),
      estimates_in_use(ATOMIC_FLAG_INIT),
      input_size(state.input_size),
      output_size(state.output_size) {
    for (auto &batch_size : state.supported_batch_sizes) {

        uint64_t estimate = 100000UL; // Default 0.1ms estimate

        // Lookup real estimate if it was provided
        auto it = state.exec_duration.find(batch_size);
        if (it != state.exec_duration.end()) {
            estimate = it->second;
        }

        // Limit the batch sizes we use
        if (batch_size == 1 || 
            (estimate > 0 
                && estimate <= scheduler->max_allowable_exec_time
                && batch_size <= scheduler->max_batch_size)) {
            if (estimates.size() < batch_size + 1) {
                estimates.resize(batch_size+1, 100000UL * Scheduler::default_clock);
            }
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

    b1_exec = estimate(1); // Use this for slo_factor
}

void Scheduler::Model::enqueue(Request request) {
    std::vector<uint64_t> batch_size_estimates(supported_batch_sizes.size());
    // Get batch size info
    for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
        batch_size_estimates[i] = estimate(supported_batch_sizes[i]);
    }

    {
        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);

        request->demand = scheduler->tracker->addRequest(
            id, batch_size_estimates[1], request->exec_slo, request->weights_slo);
    }

    // Enqueue the request
    request->id = request_id_seed_++;
    requests_queued++;
    incoming.push(request);

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

                instance->gpu->add_strategy(strategy);
            }
        }
    }
}

void Scheduler::Model::enqueue(ModelInstance* instance) {
    tbb::queuing_mutex::scoped_lock lock(mutex);

    std::vector<uint64_t> batch_size_estimates(supported_batch_sizes.size());
    for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
        batch_size_estimates[i] = estimate(supported_batch_sizes[i]);
    }

    Request newrequest;
    while (incoming.try_pop(newrequest)) queue.push_back(newrequest);
    
    for (auto &request : queue) {
        for (unsigned i = 0; i < batch_size_estimates.size(); i++) {
            auto strategy = new Scheduler::InferStrategy();
            strategy->priority = request->deadline - batch_size_estimates[i];
            strategy->deadline = request->deadline;
            strategy->request_id = request->id;
            strategy->batch_size = supported_batch_sizes[i];
            strategy->instance = instance;
            strategy->version = instance->version;

            instance->gpu->add_strategy(strategy);
        }
    }
}

void Scheduler::Model::check_timeouts(uint64_t free_at) {
    while (!queue.empty()) {
        Request request = queue.front();
        if (request->deadline >= free_at) break;

        request->set_error(clockworkControllerCouldNotStartInTime, "");
        queue.pop_front();
        requests_queued--;
        // Don't time it out here -- let the checker thread do that
    }
}

Scheduler::InferAction* Scheduler::Model::try_dequeue(
        uint64_t free_at,
        unsigned gpu_clock,
        Scheduler::InferStrategy* strategy,
        bool &retry)
{   
    // Strategy already past its deadline
    if (strategy->deadline < util::now()) {
        retry = false;
        return nullptr;
    }

    InferAction* action;
    uint64_t exec_time;
    uint64_t completion_time;
    {
        tbb::queuing_mutex::scoped_lock lock;

        if (retry) {
            if (!lock.try_acquire(mutex)) return nullptr;
        } else {
            lock.acquire(mutex);
        }
        retry = false;

        // Pull any new requests
        Request newrequest;
        while (incoming.try_pop(newrequest)) queue.push_back(newrequest);

        // Drop any timed out requests
        check_timeouts(free_at);

        // The model instance was updated since this strategy was created
        if (strategy->instance->version != strategy->version) return nullptr;

        // Pull any new requests
        while (incoming.try_pop(newrequest)) queue.push_back(newrequest);

        // Insufficient requests queued
        if (queue.size() < strategy->batch_size) return nullptr;

        // The request that generated this strategy has already completed (e.g. as part of a previous batch)
        if (queue.front()->id > strategy->request_id) return nullptr;

        // See if the strategy can actually execute given the current GPU clock
        exec_time = estimate(strategy->batch_size, gpu_clock);
        completion_time = free_at + exec_time;
        if (completion_time > strategy->deadline) return nullptr;

        // See if this strategy has enough requests to fill its batch
        //   ie. that (batch_size-1) new requests arrived after this request arrived
        // Note that this is not simply queue.size()
        unsigned available_requests = 1 + queue.back()->id - strategy->request_id;
        if (available_requests < strategy->batch_size) {

            // All is not lost; scan the queue in reverse
            available_requests = 0;
            for (auto it = queue.rbegin(); it != queue.rend(); it++) {
                if ((*it)->deadline > completion_time) break;
                available_requests++;
                if (available_requests == strategy->batch_size) {
                    // Have to inherit new deadline
                    strategy->deadline = (*it)->deadline;
                    break;
                }
            }

            // Truly insufficient requests
            if (available_requests < strategy->batch_size) return nullptr;
        }

        // Skip this request if:
        // *  a greater batch size might be achievable by a subsequent request
        // *  there is insufficient time to execute both
        unsigned candidate_batchsize = batch_lookup(available_requests);
        if (strategy->batch_size < candidate_batchsize) {
            uint64_t candidate_exec_time = estimate(candidate_batchsize, gpu_clock);
            uint64_t candidate_completion_time = free_at + candidate_exec_time;

            // We can't bump up to the candidate batch size
            if (candidate_completion_time > strategy->deadline) return nullptr;

            strategy->batch_size = candidate_batchsize;
            exec_time = candidate_exec_time;
            completion_time = candidate_completion_time;
        }

        // Drop any requests that came before this strategy and can't be included
        while (queue.size() > 0 
                && queue.front()->id != strategy->request_id
                && queue.front()->deadline < completion_time) {
            auto &request = queue.front();
            request->set_error(clockworkControllerSkipped, "");
            queue.pop_front();
            requests_queued--;
            // Don't time it out here - let the queue checker do that
        }

        // This shouldn't happen
        if (queue.size() < strategy->batch_size) return nullptr;

        action = new InferAction(scheduler, this);
        for (unsigned i = 0; i < strategy->batch_size; i++) {
            auto &request = queue.front();
            request->lock();
            action->requests.push_back(request);
            queue.pop_front();
            requests_queued--;
        }
    }
    action->set_expectations(free_at, exec_time, gpu_clock);
    action->batch();

    return action;
}


const float Scheduler::estimate_percentile = 0.99;

void Scheduler::Model::add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock) {
    while(estimates_in_use.test_and_set());

    auto it = estimators.find(batch_size);
    CHECK(it != estimators.end()) << "Unsupported batch size " << batch_size;
    auto estimator = it->second;
    estimator->insert(duration * gpu_clock);

    estimates[batch_size] = estimator->get_percentile(Scheduler::estimate_percentile);

    estimates_in_use.clear();
}

void Scheduler::Model::add_weights_measurement(uint64_t duration) {
    while (weights_in_use.test_and_set());
    weights_estimator->insert(duration);
    weights_estimate = weights_estimator->get_percentile(Scheduler::estimate_percentile);
    weights_in_use.clear();
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

Scheduler::InferAction::InferAction(Scheduler* scheduler, Model* model) : scheduler(scheduler), model(model) {
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
        size_t request_input_size = r->request.input_size;
        if (request_input_size == 0 && scheduler->generate_inputs) {
            request_input_size = model->input_size;
            generated_inputs = true;
        }
        action->input_size += request_input_size;
    }
    action->input = new char[action->input_size];
    size_t offset = 0;
    for (auto &r : requests) {
        size_t request_input_size = r->request.input_size;
        std::memcpy(action->input + offset, r->request.input, request_input_size);
        if (request_input_size == 0 && scheduler->generate_inputs) {
            offset += model->input_size;
        } else {
            offset += request_input_size;
        }
    }
}

void Scheduler::InferAction::unbatch() {
    size_t single_output_size = result->output_size / requests.size();
    if (generated_inputs) single_output_size = 0;
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
        request->set_error(error->status, error->message);
    }
}

void Scheduler::InferAction::set_result(std::shared_ptr<workerapi::InferResult> &result) {
    this->result = result;
    this->unbatch();
}

float Scheduler::InferAction::complete(uint64_t now, int gpu_id) {
    float successful_requests = 0;
    float total_requests = 0;
    for (auto &request : requests) {
        if (request->complete(now, gpu_id)) {
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
    // action->earliest = exec_start;
    // action->latest = action->earliest + Scheduler::latest_delta;
    // action->earliest = std::max(util::now() + future, exec_start - Scheduler::latest_delta);
    action->earliest = util::now();
    action->latest = std::max(util::now() + future + scheduler->latest_delta, exec_start + scheduler->latest_delta);
    // action->earliest = util::now() - Scheduler::schedule_ahead;
    // action->latest = action->expected_exec_complete + Scheduler::latest_delta;
}

        // ModelInstance* instance;
        // unsigned version;
        // ControllerActionTelemetry telemetry;
        // std::shared_ptr<workerapi::LoadWeights> action = std::make_shared<workerapi::LoadWeights>();
        // std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        // std::shared_ptr<workerapi::LoadWeightsResult> result = nullptr;

Scheduler::LoadWeightsAction::LoadWeightsAction(
    Scheduler* scheduler, ModelInstance* instance)
      : scheduler(scheduler), instance(instance) {
    action->id = action_id_seed++;
    action->model_id = instance->model->id;
}

void Scheduler::LoadWeightsAction::set_expectations(uint64_t exec_start, uint64_t duration) {
    action->expected_duration = duration;
    action->expected_exec_complete = exec_start + duration;
    action->earliest = std::max(util::now() + future, exec_start - scheduler->latest_delta);
    action->latest = std::max(util::now() + future + scheduler->latest_delta, exec_start + scheduler->latest_delta);
}

void Scheduler::LoadWeightsAction::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    if (version = instance->version) {
        instance->version++;
        instance->loading = false;
        instance->loaded = false;
    }
}

void Scheduler::LoadWeightsAction::set_result(std::shared_ptr<workerapi::LoadWeightsResult> &result) {
    this->result = result;
    if (version == instance->version) {
        instance->loaded = true;
        instance->loading = false;
    }
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

Scheduler::GPU::GPU(
    unsigned id,
    Scheduler* scheduler, 
    network::controller::WorkerConnection* worker,
    unsigned worker_id,
    unsigned gpu_id,
    unsigned pages) 
    : id(id),
      scheduler(scheduler),
      worker(worker),
      worker_id(worker_id),
      gpu_id(gpu_id),
      pages(pages),
      free_pages(pages),
      exec(Scheduler::default_clock, Scheduler::lag, Scheduler::future), 
      loadweights(Scheduler::default_clock, Scheduler::lag, Scheduler::future) {
}

void Scheduler::GPU::send_action(InferAction* action) {
    auto &infer = action->action;
    infer->gpu_id = gpu_id;
    infer->worker_id = worker_id;

    // Update GPU state
    exec.add(infer->id, infer->expected_duration);

    // Save the callback
    callbacks[infer->id] = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->infer_result(action, result);
    };
    scheduler->add_action(infer->id, this);

    // Record the telemetry
    action->telemetry.set(infer);
    action->telemetry.requests_queued = action->model->requests_queued;
    action->telemetry.copies_loaded = action->model->copies_loaded;

    // Send the action
    worker->sendAction(infer);

    // Immediately mark the requests as executing for load balancer
    {
        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);
        for (auto &request : action->requests) {
            scheduler->tracker->requestExecuting(request->demand, id);
        }
    }

    if (print_debug) std::cout << ("Worker <--  " + infer->str() + "\n");
}

void Scheduler::GPU::send_action(LoadWeightsAction* action) {
    auto &load = action->action;
    load->gpu_id = gpu_id;
    load->worker_id = worker_id;

    // Update PCI state
    loadweights.add(load->id, load->expected_duration);

    // Save the callback
    callbacks[load->id] = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->load_result(action, result);
    };
    scheduler->add_action(load->id, this);

    // Send the action
    worker->sendAction(load);

    // Record the telemetry
    action->telemetry.set(load);
    action->telemetry.requests_queued = action->instance->model->requests_queued;
    action->telemetry.copies_loaded = action->instance->model->copies_loaded;

    if (print_debug || print_loads) std::cout << ("Worker <--  " + load->str() + "\n");
}

void Scheduler::GPU::send_action(EvictWeightsAction* action) {
    auto &evict = action->action;
    evict->gpu_id = gpu_id;
    evict->worker_id = worker_id;

    // Save the callback
    callbacks[evict->id] = [this, action](std::shared_ptr<workerapi::Result> result) {
        this->evict_result(action, result);
    };
    scheduler->add_action(evict->id, this);

    // Send the action
    worker->sendAction(evict);

    // Record the telemetry
    action->telemetry.set(evict);
    action->telemetry.requests_queued = action->instance->model->requests_queued;
    action->telemetry.copies_loaded = action->instance->model->copies_loaded+1;

    if (print_debug || print_loads) std::cout << ("Worker <--  " + evict->str() + "\n");    
}

std::vector<Scheduler::EvictWeightsAction*> Scheduler::GPU::evict_pages(unsigned required_pages) {
    std::vector<EvictWeightsAction*> ret;
    while (free_pages < required_pages) {
        int model_id = scheduler->tracker->evictModel(id);

        if (model_id == -1) break;

        ModelInstance* instance = instances[model_id];
        instance->version++;
        instance->loading = false;
        instance->loaded = false;
        instance->model->copies_loaded--;

        EvictWeightsAction* evict = new EvictWeightsAction(instance);
        evict->set_expectations();
        ret.push_back(evict);
        
        free_pages += scheduler->models[model_id]->num_weights_pages;
        eviction_required = true; // GPU reached capacity; evictions required in future
    }
    return ret;
}

bool Scheduler::GPU::try_load(uint64_t available) {
    uint64_t now = util::now();
    if (last_load + 100000UL > now) return false;

    ModelInstance* instance;
    unsigned size;
    std::vector<EvictWeightsAction*> evict_actions;
    {
        tbb::queuing_mutex::scoped_lock lock;

        // if (!lock.try_acquire(scheduler->tracker->mutex)) return false;
        lock.acquire(scheduler->tracker->mutex);

        last_load = now;


        int model_id = scheduler->tracker->loadModel(id, eviction_required);
        if (model_id == -1) {
            return false;
        }

        instance = instances[model_id];
        CHECK(instance->loaded == false && instance->loading == false) << "Tracker asked to load model that is already loaded";

        size = scheduler->models[model_id]->num_weights_pages;
        evict_actions = evict_pages(size);

        if (free_pages < size) {
            scheduler->tracker->loadModelComplete(id, model_id, false);
        }
    }

    // Send the evict actions
    for (auto &evict : evict_actions) {
        send_action(evict);
    }

    if (free_pages < size) {
        return false;
    }

    instance->version++;
    instance->loading = true;
    instance->loaded = false;
    free_pages -= size;

    uint64_t expected_duration = instance->model->estimate_weights();

    LoadWeightsAction* action = new LoadWeightsAction(scheduler, instance);
    action->version = instance->version;
    action->set_expectations(available, expected_duration);
    loads++;

    // uint64_t exec_may_begin_at = std::max(available, util::now() + latest_delta) + expected_duration;
    // loading.push({instance, instance->version, exec_may_begin_at});

    send_action(action);
    return true;
}

void Scheduler::GPU::check_pending() {
    bool active = false;

    // Handle all incoming results
    std::shared_ptr<workerapi::Result> result;
    while (results.try_pop(result)) {
        handle_result(result);
        active = true;
    }

    uint64_t exec_at = exec.available();

    // // See if any model instances are at the point where their actions can be scheduled
    // while (loading.size() > 0) {
    //     auto &load = loading.front();

    //     if (load.version == load.instance->version && (load.instance->loading || load.instance->loaded)) {
    //         if (load.available_at > exec_at) break;

    //         // Ready to enqueue infer strategies for this model
    //         load.instance->model->enqueue(load.instance);
    //         active = true;
    //     }
    //     loading.pop();
    // }

    // Drain all incoming strategies, add to priority queue
    InferStrategy* strategy;
    while (strategies.try_pop(strategy)) {
        queue.push(strategy);
        active = true;
    }

    // Drop all strategies that have passed
    uint64_t now = util::now();
    while (queue.size() > 0) {
        auto strategy = queue.top();
        if (strategy->deadline > now) break;
        queue.pop();
        delete strategy;
    }

    // Schedule infer actions
    uint64_t schedule_until = now + scheduler->schedule_ahead;
    while ((exec_at = exec.available()) < schedule_until && queue.size() > 0) {
        InferStrategy* strategy = queue.top();

        // bool retry = last_exec + 200000UL > util::now();
        bool retry = false;
        InferAction* action = strategy->instance->model->try_dequeue(exec_at, exec.clock(), strategy, retry);

        if (retry) break;

        queue.pop();

        delete strategy;

        if (action != nullptr) {
            send_action(action);
            last_exec = util::now();
            active = true;
        }
    }

    // Schedule one load action
    uint64_t load_at;
    if (loads < Scheduler::max_loads && (load_at = loadweights.available()) < schedule_until) {
        if (try_load(load_at)) {
            active = true;
        }
    }

    if (!active) {
        usleep(50);
    }
}

void Scheduler::GPU::handle_result(std::shared_ptr<workerapi::Result> result) {
    auto it = callbacks.find(result->id);
    CHECK(it != callbacks.end()) 
        << "Received result for non-existent action " << result->str();

    auto callback = it->second;
    callbacks.erase(it);

    callback(result);
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
    CHECK(action->complete(util::now(), id) == 0) << "ErrorResult should not result in successful requests";

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
    action->telemetry.goodput = action->complete(util::now(), id);

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
    {
        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);
        scheduler->tracker->loadModelComplete(id, action->instance->model->id, false);
    }
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
    {
        tbb::queuing_mutex::scoped_lock lock(scheduler->tracker->mutex);
        scheduler->tracker->loadModelComplete(id, action->instance->model->id, true);
    }

    // Update PCI state tracking
    loadweights.success(result->id, result->end);

    // Update PCI tracking
    action->instance->model->add_weights_measurement(result->duration);
    action->instance->model->copies_loaded++;

    // Enable new requests
    action->instance->model->enqueue(action->instance);

    // TODO: change this?
    action->telemetry.goodput = 1.0;

    scheduler->printer->log(action->telemetry);

    delete action;
}

void Scheduler::GPU::load_result(LoadWeightsAction* action, std::shared_ptr<workerapi::Result> &result) {
    if (!print_debug && print_loads) std::cout << ("Worker  --> " + result->str() + "\n");

    loads--;

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
        gpu_pages += gpu->pages;
    }


    std::cout << "Total GPU capacity " << gpu_pages << " pages (" << gpu_pages/gpus.size() << " per GPU)." << std::endl
              << "Total model pages " << model_pages << " (" << (100*model_pages / gpu_pages) << "% oversubscription)." << std::endl;
}

void Scheduler::initialize_models(ClockworkState &state) {
    models.resize(state.workers[0].models.size(), nullptr);

    for (auto &p : state.workers[0].models) {
        auto &model = p.second;

        if (model.id >= models.size()) {
            models.resize(model.id+1, nullptr);
        }

        models[model.id] = new Model(this, model);
    }

    std::cout << "Created " << models.size() << " models" << std::endl;
}

void Scheduler::initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                ClockworkState &state) 
{
    unsigned total_pages = 0;
    unsigned workers_remaining = state.workers.size();
    unsigned gpus_remaining = max_gpus;
    for (WorkerState &worker : state.workers) {
        int num_gpus = std::min((unsigned) worker.gpus.size(), gpus_remaining / workers_remaining);
        for (unsigned i = 0; i < num_gpus; i++) {
            GPUState &gpustate = worker.gpus[i];
            GPU* gpu = new GPU(
                gpus.size(),
                this,
                workers[worker.id],
                worker.id,
                gpustate.id,
                gpustate.weights_cache_total_pages
            );
            gpus.push_back(gpu);

            total_pages += gpu->pages;
        }
        gpus_remaining -= num_gpus;
        workers_remaining--;
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

    tracker = new WorkTracker2(gpus.size(), models.size(), default_slo);
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

    // Create and start the printer thread
    this->printer = ControllerActionTelemetry::log_and_summarize(actions_filename, print_interval);

    // Create and start the GPU threads
    for (unsigned i = 0; i < gpus.size(); i++) {
        std::vector<GPU*> gpus_for_thread = {gpus[i]};
        gpu_threads.push_back(std::thread(&Scheduler::run_gpu_thread, this, gpus_for_thread));
        threading::initLowPriorityThread(gpu_threads[i]);
    }
}

void Scheduler::handle_request(Request request) {
    // Enqueue the request to the model
    unsigned model_id = request->request.model_id;
    if (model_id > models.size() || models[model_id] == nullptr) {
        request->set_error(clockworkError, "Invalid model ID");
        CHECK(!request->complete(util::now(), -1)) << "Erroneous request should not be successful";
        return;
    }

    Model* model = models[model_id];
    request->set_model(model);
    request->set_slo(default_slo);
    model->enqueue(request);
    requests.push(request);
}

void Scheduler::add_action(uint64_t action_id, GPU* gpu) {
    auto pair = std::make_pair(action_id, gpu);

    while (actions_in_use.test_and_set());
    outstanding_actions.insert(pair);
    actions_in_use.clear();
}

void Scheduler::handle_result(std::shared_ptr<workerapi::Result> result) {
    while (actions_in_use.test_and_set());

    auto it = outstanding_actions.find(result->id);
    CHECK(it != outstanding_actions.end()) 
        << "Received result for non-existent action " << result->str();

    auto gpu = it->second;
    outstanding_actions.erase(it);

    actions_in_use.clear();

    gpu->add_result(result);
}

void Scheduler::run() {

    // Start processing actions + results
    while (true) {
        Request request;
        while (request_queue.try_pop(request)) {
            handle_request(request);
        }

        // Drop any timed out requests
        uint64_t now = util::now();
        while (!requests.empty()) {
            auto &request = requests.front();

            if (request->deadline > now) break;

            request->finalize();
            requests.pop();
        }
    }
}

void Scheduler::run_gpu_thread(std::vector<GPU*> gpus) {
    std::stringstream msg;
    msg << "GPU handler [ ";
    for (auto &gpu : gpus) {
        msg << gpu->id << " ";
    }
    msg << "] started" << std::endl;
    std::cout << msg.str();

    while (true) {
        for (auto &gpu : gpus) {
            gpu->check_pending();
        }
    }
}

// The actual scheduler interface implementation, invoked by worker network thread
void Scheduler::resultFromWorker(std::shared_ptr<workerapi::Result> result)
{
    if (print_debug) std::cout << ("Worker  --> " + result->str() + "\n");

    result->result_received = util::now();
    handle_result(result);
}

// The actual scheduler interface implementation, invoked by client network thread
void Scheduler::clientInfer(clientapi::InferenceRequest &request, 
    std::function<void(clientapi::InferenceResponse&)> callback)
{
    if (print_debug) std::cout << ("Client  --> " + request.str() + "\n");

    request_queue.push(std::make_shared<RequestImpl>(this, request, callback));
}

}
}
}