#include "clockwork/controller/infer_only_scheduler.h"

namespace clockwork {

uint64_t action_id_seed = 0;

InferOnlyScheduler::InferOnlyScheduler(std::string actions_filename) 
    : actions_filename(actions_filename) {
}

InferOnlyScheduler::Request::Request(clientapi::InferenceRequest request,
    std::function<void(clientapi::InferenceResponse&)> callback) : 
        request(request), 
        callback(callback), 
        deadline(util::now() + InferOnlyScheduler::slo) {
    response.header.user_request_id = request.header.user_request_id;
    response.header.message = "";
    response.model_id = request.model_id;
    response.batch_size = request.batch_size;
    response.output = nullptr;
    response.output_size = 0;
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

void InferOnlyScheduler::Request::complete() {
    if (print_debug) std::cout << ("Client <--  " + response.str() + "\n");

    callback(response);
}

std::vector<unsigned> batch_lookup = {0,1,2,2,4,4,4,4,8,8,8,8,8,8,8,8,16};

InferOnlyScheduler::Model::Model(unsigned id) : id(id) {}

void InferOnlyScheduler::Model::enqueue(Request* request) {
    request->id = request_id_seed++;
    queue.push(request);
}

void InferOnlyScheduler::Model::check_timeouts() {
    uint64_t now = util::now();
    while (!queue.empty() && queue.front()->deadline < now) {
        Request* request = queue.front();
        queue.pop();

        request->response.header.status = clockworkTimeout;
        request->complete();                
        delete request;
    }
    if (queue.empty()) assigned_gpu = nullptr;
}

InferOnlyScheduler::Action* InferOnlyScheduler::Model::try_dequeue(uint64_t expected_request_id) {
    if (queue.empty()) return nullptr;
    if (queue.front()->id > expected_request_id) return nullptr;

    check_timeouts();

    unsigned batch_size = batch_lookup[std::min(batch_lookup.size()-1, queue.size())];
    if (batch_size == 0) return nullptr;

    Action* action = new Action(id);
    for (unsigned i = 0; i < batch_size; i++) {
        action->requests.push_back(queue.front());
        queue.pop();
    }
    action->batch();

    if (queue.empty()) assigned_gpu = nullptr;

    return action;
}


InferOnlyScheduler::Action::Action(unsigned model_id) {
    action->id = action_id_seed++;
    action->model_id = model_id;
    action->earliest = util::now();
    action->latest = action->earliest + 100000000UL; // 100 ms
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

void InferOnlyScheduler::Action::complete() {
    for (Request* request : requests) {
        request->complete();
        delete request;
    }
}

void InferOnlyScheduler::GPU::send_action(Action* action) {
    auto &infer = action->action;
    infer->gpu_id = gpu_id;

    action->telemetry.set(infer);
    action->telemetry.worker_id = worker_id;

    // Save it and send
    this->outstanding++;
    scheduler->outstanding_actions[infer->id] = {this, action};
    worker->sendAction(infer);

    if (print_debug) std::cout << ("Worker <--  " + infer->str() + "\n");
}

void InferOnlyScheduler::GPU::check_pending() {
    while (outstanding < max_outstanding && scheduler->queue.size() > 0) {
        auto &next = scheduler->queue.front();
        scheduler->queue.pop();

        Action* action = next.model->try_dequeue(next.request_id);
        if (action != nullptr) {
            send_action(action);
        }
    }
}

void InferOnlyScheduler::GPU::handle_error(Action* action, std::shared_ptr<workerapi::ErrorResult> &error) {
    action->telemetry.set(error);
    outstanding--;

    check_pending();

    action->set_error(error);
    action->complete();

    scheduler->printer->log(action->telemetry);

    delete action;
}

void InferOnlyScheduler::GPU::handle_success(Action* action, std::shared_ptr<workerapi::InferResult> &result) {
    action->telemetry.set(result);
    outstanding--;

    check_pending();

    action->set_result(result);
    action->complete();

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

            this->models[model.id] = new Model(model.id);
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
            gpu_fifo.push(gpu);
        }
    }
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
        request->complete();
        delete request;
        return;
    }

    Model* model = it->second;
    model->enqueue(request);

    if (model->assigned_gpu == nullptr) {
        model->assigned_gpu = gpu_fifo.front();
        gpu_fifo.pop();
        gpu_fifo.push(model->assigned_gpu);
    }


    queue.push(QueueElement{request->id, model});
    model->assigned_gpu->check_pending();
}

void InferOnlyScheduler::handle_result(std::shared_ptr<workerapi::Result> result) {
    auto it = outstanding_actions.find(result->id);
    CHECK(it != outstanding_actions.end()) 
        << "Received result for non-existent action " << result->str();

    OutstandingAction o = it->second;
    outstanding_actions.erase(it);

    o.gpu->handle_result(o.action, result);
    o.gpu->check_pending();
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
    while (true) {
        Request* request;
        if (request_queue.try_pop(request)) {
            handle_request(request);
        }

        std::shared_ptr<workerapi::Result> result;
        if (result_queue.try_pop(result)) {
            handle_result(result);
        }
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