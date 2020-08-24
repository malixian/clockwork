#include "clockwork/controller/infer_only_scheduler.h"

namespace clockwork {

uint64_t action_id_seed = 0;

InferOnlyScheduler::InferOnlyScheduler(std::string actions_filename) 
    : actions_filename(actions_filename) {
}

InferOnlyScheduler::Request::Request(clientapi::InferenceRequest request,
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
    response.deadline = request.arrival + InferOnlyScheduler::slo;

    this->deadline = response.deadline - InferOnlyScheduler::buffer;
}

InferOnlyScheduler::Request::~Request() {
    delete static_cast<char*>(request.input);
}

void InferOnlyScheduler::Request::set_result(char* output, size_t output_size) {
    response.header.status = clockworkSuccess;
    response.output = output;
    response.output_size = output_size;
}

void InferOnlyScheduler::Request::set_error(int status, std::string message) {
    response.header.status = status;
    response.header.message = message;
}

bool InferOnlyScheduler::Request::complete(uint64_t now) {
    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    // Set the departure time (controller.cpp can also do this, 
    // but we want to report departure time back to the action to determine goodput)
    this->departure = now;
    response.departure = now;

    callback(response);

    return response.header.status == clockworkSuccess && departure <= response.deadline;
}

void InferOnlyScheduler::Request::timeout(uint64_t now) {
    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    response.header.status = clockworkTimeout;
    response.departure = now;

    callback(response);
}

std::vector<unsigned> batch_lookup = {0,1,2,2,4,4,4,4,8,8,8,8,8,8,8,8,16};

InferOnlyScheduler::Model::Model(BatchedModelState &state) : id(state.id) {
    // Store initial exec estimates
    for (auto &batch_size : state.supported_batch_sizes) {
        auto it = state.exec_duration.find(batch_size);
        if (it == state.exec_duration.end()) {
            estimates[batch_size] = 1000000UL; // Start with 0.1ms estimate
        } else {
            estimates[batch_size] = it->second * InferOnlyScheduler::default_clock;
        }
        estimators[batch_size] = new SlidingWindow(InferOnlyScheduler::estimate_window_size);
    }
}

void InferOnlyScheduler::Model::enqueue(Request* request) {
    request->id = request_id_seed++;
    queue.push(request);
}

void InferOnlyScheduler::Model::check_timeouts(GPU* gpu, uint64_t now) {
    while (!queue.empty()) {
        Request* request = queue.front();
        if (request->deadline >= now + estimate(1, gpu->tracker.clock())) break;

        queue.pop();
        request->timeout(now);
        delete request;
    }
}

InferOnlyScheduler::Action* InferOnlyScheduler::Model::try_dequeue(GPU* gpu, uint64_t free_at, uint64_t expected_request_id) {
    if (queue.empty()) return nullptr;
    if (queue.front()->id > expected_request_id) return nullptr;

    check_timeouts(gpu, free_at);

    unsigned max_batch_size = batch_lookup[std::min(batch_lookup.size()-1, queue.size())];
    if (max_batch_size == 0) return nullptr;

    uint64_t batch_deadline = queue.front()->deadline;

    int clock = gpu->tracker.clock();
    unsigned batch_size = 1;
    Action* action = new Action(this);
    while (true) {
        while (action->requests.size() < batch_size) {
            action->requests.push_back(queue.front());
            queue.pop();
        }

        if (batch_size * 2 > max_batch_size) break;

        uint64_t exec = estimate(batch_size * 2, clock);
        if (free_at + exec > batch_deadline) break;

        batch_size *= 2;
    }
    action->set_expectations(free_at, estimate(batch_size, clock), clock);
    action->batch();

    return action;
}


const float InferOnlyScheduler::estimate_percentile = 0.99;

void InferOnlyScheduler::Model::add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock) {
    auto it = estimators.find(batch_size);
    CHECK(it != estimators.end()) << "Unsupported batch size " << batch_size;
    auto estimator = it->second;
    estimator->insert(duration * gpu_clock);

    estimates[batch_size] = estimator->get_percentile(InferOnlyScheduler::estimate_percentile);
}

uint64_t InferOnlyScheduler::Model::estimate(unsigned batch_size, int clock) {
    unsigned effective_batch_size = batch_lookup[batch_size];
    return estimates[effective_batch_size] / clock;
}

InferOnlyScheduler::Action::Action(Model* model) : model(model) {
    action->id = action_id_seed++;
    action->model_id = model->id;
    action->earliest = util::now();
}

InferOnlyScheduler::Action::~Action() {
    if (result != nullptr) {
        delete result->output;
        delete action->input;
    }
}

void InferOnlyScheduler::Action::batch() {
    action->batch_size = requests.size();
    action->input_size = 0;
    for (Request* r : requests) {
        action->input_size += r->request.input_size;
    }
    action->input = new char[action->input_size];
    size_t offset = 0;
    for (Request* r : requests) {
        std::memcpy(action->input + offset, r->request.input, r->request.input_size);
        offset += r->request.input_size;
    }
}

void InferOnlyScheduler::Action::unbatch() {
    size_t single_output_size = result->output_size / requests.size();
    size_t offset = 0;
    for (unsigned i = 0; i < requests.size(); i++) {
        char* output = new char[single_output_size];
        std::memcpy(output, result->output + offset, single_output_size);
        offset += single_output_size;

        requests[i]->set_result(output, single_output_size);
    }
}

void InferOnlyScheduler::Action::set_error(std::shared_ptr<workerapi::ErrorResult> &error) {
    this->error = error;
    for (Request* request : requests) {
        request->set_error(clockworkError, error->message);
    }
}

void InferOnlyScheduler::Action::set_result(std::shared_ptr<workerapi::InferResult> &result) {
    this->result = result;
    this->unbatch();
}

float InferOnlyScheduler::Action::complete(uint64_t now) {
    float successful_requests = 0;
    float total_requests = 0;
    for (Request* request : requests) {
        if (request->complete(now)) {
            successful_requests += 1;
        }
        total_requests += 1;
        delete request;
    }
    return successful_requests / total_requests;
}

void InferOnlyScheduler::Action::set_expectations(uint64_t exec_start, uint64_t duration, int clock) {
    action->expected_duration = duration;
    action->expected_exec_complete = exec_start + duration;
    action->expected_gpu_clock = clock;
    action->earliest = util::now();
    action->latest = std::max(action->expected_exec_complete, action->earliest+InferOnlyScheduler::schedule_ahead);
}

InferOnlyScheduler::GPU::GPU() : tracker(InferOnlyScheduler::default_clock) {
}

void InferOnlyScheduler::GPU::send_action(Action* action) {
    auto &infer = action->action;
    infer->gpu_id = gpu_id;
    infer->worker_id = worker_id;

    action->telemetry.set(infer);

    // Update GPU state
    tracker.add(infer->id, infer->expected_duration);

    // Save it and send
    scheduler->outstanding_actions[infer->id] = {this, action};
    worker->sendAction(infer);

    if (print_debug) std::cout << ("Worker <--  " + infer->str() + "\n");
}

void InferOnlyScheduler::GPU::check_pending() {
    uint64_t schedule_until = util::now() + schedule_ahead;
    uint64_t available;
    while ((available = tracker.available()) < schedule_until && scheduler->queue.size() > 0) {
        auto &next = scheduler->queue.front();
        scheduler->queue.pop();

        Action* action = next.model->try_dequeue(this, available, next.request_id);
        if (action != nullptr) {
            send_action(action);
        }
    }
}

void InferOnlyScheduler::GPU::handle_error(Action* action, std::shared_ptr<workerapi::ErrorResult> &error) {
    std::cout << ("Worker  --> " + error->str() + "\n");

    action->telemetry.set(error);
    
    // Update GPU state tracking
    tracker.error(error->id, util::now());

    action->set_error(error);
    CHECK(action->complete(util::now()) == 0) << "ErrorResult should not result in successful requests";

    scheduler->printer->log(action->telemetry);

    delete action;
}

void InferOnlyScheduler::GPU::handle_success(Action* action, std::shared_ptr<workerapi::InferResult> &result) {
    action->telemetry.set(result);

    // Update GPU state tracking
    tracker.success(result->id, result->exec.end);
    tracker.update_clock(result->gpu_clock);

    // Update model execution tracking
    action->model->add_measurement(action->action->batch_size, result->exec.duration, result->gpu_clock);

    action->set_result(result);
    action->telemetry.goodput = action->complete(util::now());

    scheduler->printer->log(action->telemetry);

    delete action;
}

void InferOnlyScheduler::GPU::handle_result(Action* action, std::shared_ptr<workerapi::Result> &result) {
    if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
        handle_error(action, error);

    } else if (auto infer = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
        handle_success(action, infer);

    } else {
        CHECK(false) << "Unexpected response to Infer action" << result->str();

    }
}

unsigned InferOnlyScheduler::GPU::load_model_weights() {
    unsigned action_count = 0;
    for (auto &p : scheduler->models) {
        auto load = std::make_shared<workerapi::LoadWeights>();
        load->id = action_id_seed++;
        load->gpu_id = gpu_id;
        load->model_id = p.first;
        load->earliest = 0;
        load->latest = UINT64_MAX;

        worker->sendAction(load);
        action_count++;
    }
    return action_count;
}

void InferOnlyScheduler::validate_clockwork_state(ClockworkState &state) {
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

void InferOnlyScheduler::initialize_models(ClockworkState &state) {
    unsigned max_pages = state.workers[0].gpus[0].weights_cache_total_pages;

    unsigned total = 0;
    std::vector<unsigned> models_to_load;
    for (auto &p : state.workers[0].models) {
        auto &model = p.second;
        if (total + model.num_weights_pages < max_pages) {
            total += model.num_weights_pages;

            this->models[model.id] = new Model(model);
        }
    }

    std::cout << "Created " << this->models.size() << " models using " 
              << total << "/" << max_pages << " pages (" 
              << (total*100)/max_pages << "%)" << std::endl;
}

void InferOnlyScheduler::initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                ClockworkState &state) 
{
    for (WorkerState &worker : state.workers) {
        for (GPUState &gpustate : worker.gpus) {
            GPU* gpu = new GPU();
            gpu->scheduler = this;
            gpu->worker = workers[worker.id];
            gpu->worker_id = worker.id;
            gpu->gpu_id = gpustate.id;
            gpus.push_back(gpu);
            std::cout << "Create Worker-" << gpu->worker_id << " GPU-" << gpu->gpu_id << std::endl;
        }
    }
    std::cout << "Created " << gpus.size() << " Worker-GPUs" << std::endl;
}


// Called when model loading has completed
void InferOnlyScheduler::start(std::vector<network::controller::WorkerConnection*> workers,
                    ClockworkState &state) 
{
    validate_clockwork_state(state);
    initialize_models(state);
    initialize_gpus(workers, state);

    this->thread = std::thread(&InferOnlyScheduler::run, this);
    threading::initHighPriorityThread(this->thread);
}

void InferOnlyScheduler::handle_request(Request* request) {
    // Enqueue the request to the model
    auto it = models.find(request->request.model_id);
    if (it == models.end()) {
        request->set_error(clockworkError, "Invalid model ID");
        CHECK(!request->complete(util::now())) << "Erroneous request should not be successful";
        delete request;
        return;
    }

    Model* model = it->second;
    model->enqueue(request);

    queue.push(QueueElement{request->id, model});
}

void InferOnlyScheduler::handle_result(std::shared_ptr<workerapi::Result> result) {
    auto it = outstanding_actions.find(result->id);
    CHECK(it != outstanding_actions.end()) 
        << "Received result for non-existent action " << result->str();

    OutstandingAction o = it->second;
    outstanding_actions.erase(it);

    o.gpu->handle_result(o.action, result);
}

void InferOnlyScheduler::run() {
    // Load weights for all models
    unsigned outstanding_loads = 0;
    for (GPU* gpu : gpus) {
        outstanding_loads += gpu->load_model_weights();
    }

    // Wait for the weights to load
    while (outstanding_loads > 0) {
        std::shared_ptr<workerapi::Result> result;
        while (!result_queue.try_pop(result)) ;

        if (auto load = std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(result)) {
            outstanding_loads--;
        } else {
            CHECK(false) << "InferOnlyScheduler error during model load phase: " 
                         << result->str();
        }
    }

    // Create and start the printer thread
    printer = ControllerActionTelemetry::log_and_summarize(actions_filename, print_interval);

    // Start processing actions + results
    unsigned start = 0;
    while (true) {
        Request* request;
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
    }
}

// The actual scheduler interface implementation, invoked by worker network thread
void InferOnlyScheduler::resultFromWorker(std::shared_ptr<workerapi::Result> result)
{
    if (print_debug) std::cout << ("Worker  --> " + result->str() + "\n");

    result_queue.push(result);
}

// The actual scheduler interface implementation, invoked by client network thread
void InferOnlyScheduler::clientInfer(clientapi::InferenceRequest &request, 
    std::function<void(clientapi::InferenceResponse&)> callback)
{
    if (print_debug) std::cout << ("Client  --> " + request.str() + "\n");

    request_queue.push(new Request(request, callback));
}

}