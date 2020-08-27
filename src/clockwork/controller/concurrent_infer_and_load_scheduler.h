// Copyright 2020 Max Planck Institute for Software Systems

#ifndef SRC_CLOCKWORK_CONTROLLER_CONCURRENT_INFER_AND_LOAD_SCHEDULER_H_
#define SRC_CLOCKWORK_CONTROLLER_CONCURRENT_INFER_AND_LOAD_SCHEDULER_H_

#include <atomic>
#include <algorithm>
#include <string>
#include <sstream>
#include <set>
#include "clockwork/controller/scheduler.h"
#include "clockwork/controller/worker_tracker.h"
#include "clockwork/controller/load_tracker.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include "clockwork/thread.h"
#include "clockwork/api/worker_api.h"
#include "clockwork/sliding_window.h"
#include "tbb/mutex.h"
#include "tbb/queuing_mutex.h"

namespace clockwork {
namespace scheduler {
namespace infer4 {

class Scheduler : public clockwork::Scheduler {
 public:

    static const uint64_t print_interval = 10000000000UL;
    static const bool print_debug = false;
    static const bool print_loads = false;

    // Non-configurable parameters
    static const uint64_t default_clock = 1380; // default gpu clock speed
    static const uint64_t buffer = 5000000UL; // Aim for an SLO this much prior to actual SLO
    static const int estimate_window_size = 10; // Estimate execution time using last 10 measurements
    static const float estimate_percentile; // Percentile to use for estimation; 0.99 (effectively max)
    static const uint64_t lag = 10000000UL; // how much can worker lag behind expected completion time before we stop scheduling
    static const uint64_t future = 1000000UL; // used for setting earliest timestamp; expect 1ms lag getting to worker
    static const int max_loads = 2; // max number of outstanding loads
    static const uint64_t max_loadweights_slo = 25000000UL;

    // Scheduler parameters configurable by ./controller binary

    const uint64_t default_slo;
    const uint64_t latest_delta; // Actions can run up to 10ms behind schedule before the worker will drop them
    const uint64_t schedule_ahead; // schedule 10ms into the future
    const uint64_t max_allowable_exec_time; // disallow batches with execution times greater than this
    const unsigned max_batch_size;
    const bool generate_inputs; // if clients send 0-size inputs, do we want to generate real ones, or send 0-size?
    const int max_gpus; // max number of gpus to use

    Scheduler(
        uint64_t default_slo, // 100ms
        uint64_t latest_delta, // 10ms
        uint64_t schedule_ahead, // 10ms
        bool generate_inputs, // if clients send no input, should we generate real inputs, or forward the size-0?
        int max_gpus, // max GPUs to use
        uint64_t max_allowable_exec_time = 18000000UL, // 18ms
        unsigned max_batch_size = 8, // max supported batchsize of 8
        std::string actions_filename = "/local/clockwork_action_log.tsv");

    class StrategyImpl;
    typedef std::shared_ptr<StrategyImpl> Strategy;

    class RequestImpl;
    typedef std::shared_ptr<RequestImpl> Request;

    class Model;
    class RequestImpl {
     public:
        Scheduler* scheduler;
        uint64_t id;
        uint64_t slo;
        uint64_t exec_slo;
        uint64_t weights_slo;
        uint64_t deadline;
        Model* model = nullptr;
        clientapi::InferenceRequest request;
        clientapi::InferenceResponse response;

        LoadTracker::Demand demand;
        std::vector<Strategy> strategies;

     private:
        std::atomic_bool locked;
        std::atomic_flag response_sent;

        std::function<void(clientapi::InferenceResponse&)> callback;

     public:
        RequestImpl(Scheduler* scheduler,
            clientapi::InferenceRequest request,
            std::function<void(clientapi::InferenceResponse&)> callback);
        ~RequestImpl();

        void set_model(Model* model);
        void set_slo(uint64_t default_slo);
        void set_result(char* output, size_t output_size);
        void set_error(int status, std::string message);

        void invalidate_strategies() {
            for (auto &strategy : strategies) {
                strategy->valid = false;
            }
        }

        void lock();

        // Returns true if the result was successful and within the deadline
        void timeout();
        bool complete(uint64_t now, int gpu_id);
        void finalize();

        struct DeadlineComparator {
            bool operator()(const Request &lhs, const Request &rhs) {
                return lhs->deadline > rhs->deadline;
            }
        };
    };

    class InferAction {
    private:
        Scheduler* scheduler;
        bool generated_inputs = false;
     public:
        Model* model;
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::Infer> action = std::make_shared<workerapi::Infer>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::InferResult> result = nullptr;
        std::vector<Request> requests;

        explicit InferAction(Scheduler* scheduler, Model* model);
        ~InferAction();

        void batch();
        void unbatch();
        void set_expectations(uint64_t exec_start, uint64_t duration, int clock);
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::InferResult> &result);

        // Returns the fraction of successful requests
        float complete(uint64_t now, int gpu_id);
    };

    class ModelInstance;
    class LoadWeightsAction {
     public:
        Scheduler* scheduler;
        ModelInstance* instance;
        unsigned version;
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::LoadWeights> action = std::make_shared<workerapi::LoadWeights>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::LoadWeightsResult> result = nullptr;

        explicit LoadWeightsAction(Scheduler* scheduler, ModelInstance* instance);

        void set_expectations(uint64_t exec_start, uint64_t duration);
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::LoadWeightsResult> &result);

    };
    class EvictWeightsAction {
     public:
        ModelInstance* instance;
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::EvictWeights> action = std::make_shared<workerapi::EvictWeights>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::EvictWeightsResult> result = nullptr;

        explicit EvictWeightsAction(ModelInstance* instance);

        void set_expectations();
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::EvictWeightsResult> &result);

    };

    class GPU;

    class ModelInstance {
     public:
        GPU* gpu = nullptr;
        Model* model = nullptr;
        std::atomic_bool loaded;
        std::atomic_bool loading;
        unsigned version = 0;
        ModelInstance(GPU* gpu, Model* model): gpu(gpu), model(model), loaded(false), loading(false) {}
    };

    class Model {
     public:
        unsigned id;
        Scheduler* scheduler;
        unsigned num_weights_pages;
        size_t input_size;
        size_t output_size;
        std::vector<ModelInstance*> instances;
        uint64_t b1_exec;
        std::atomic_int copies_loaded = 0;
        std::atomic_int requests_queued = 0;

     private:
        tbb::queuing_mutex mutex;

        std::vector<unsigned> supported_batch_sizes;
        std::vector<unsigned> batch_lookup_;
        unsigned max_batch_size;

        std::atomic_flag estimates_in_use;
        std::vector<uint64_t> estimates;
        std::map<unsigned, SlidingWindow*> estimators;

        std::atomic_flag weights_in_use;
        SlidingWindow* weights_estimator;
        uint64_t weights_estimate;

        std::atomic_uint64_t request_id_seed_ = 0;

        tbb::concurrent_queue<Request> incoming;
        std::deque<Request> queue;

     public:

        Model(Scheduler* scheduler, BatchedModelState &state);

        uint64_t estimate_weights();

        // Enqueues the request to this model, then enqueues InferStrategies to all active ModelInstances
        void enqueue(Request request);

        // Enqueues InferStrategies for all requests to this ModelInstance
        void enqueue(ModelInstance* instance);

        // Gets actions to execute for this model
        InferAction* try_dequeue(uint64_t gpu_free_at, unsigned gpu_clock, Strategy &strategy, bool &retry);

        // GPUs can add new measurements
        void add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock);
        void add_weights_measurement(uint64_t duration);
        uint64_t estimate(unsigned batch_size);

     private:

        // For num_requests requests, what is the maximum batch size we could execute?
        unsigned batch_lookup(unsigned num_requests);

        void check_timeouts(uint64_t free_at);
        uint64_t estimate(unsigned batch_size, int clock);
    };

    class StrategyImpl {
    public:
        std::atomic_bool valid;

        uint64_t priority;
        uint64_t deadline;
        uint64_t request_id;
        unsigned batch_size;

        Model* model;

        struct Comparator {
            bool operator()(const Strategy &lhs, const Strategy &rhs) {
                return lhs->priority > rhs->priority;
            }
        };

        std::string str() {
            std::stringstream ss;
            ss << "S p=" << priority << " d=" << deadline << " rid=" << request_id << " b=" << batch_size;
            return ss.str();
        }

        StrategyImpl() : valid(true) {}

    };

    struct Loading {
        ModelInstance* instance;
        unsigned version;
        uint64_t available_at;
    };

    class GPU {
     public:
        unsigned id; // a unique id for this (worker_id, gpu_id)
        unsigned worker_id; // the id of the worker
        unsigned gpu_id; // the id of the gpu on the worker
        unsigned pages; // the number of pages on the gpu
        std::vector<ModelInstance*> instances;

     private:
        network::controller::WorkerConnection* worker;
        Scheduler* scheduler;
        WorkerTracker exec;
        WorkerTracker loadweights;

        unsigned free_pages;
        bool eviction_required = false;
        uint64_t last_load = 0;
        uint64_t last_exec = 0;
        uint64_t last_print = 0;
        int loads = 0;

        std::queue<Loading> loading;
        std::priority_queue<Strategy, std::deque<Strategy>, StrategyImpl::Comparator> strategy_queue;

        // Incoming strategies enqueued by models
        tbb::concurrent_queue<Request> incoming_strategies;

        // Results enqueued by network
        tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> incoming_results;

        // Pending actions on this GPU
        typedef std::function<void(std::shared_ptr<workerapi::Result>&)> action_callback;
        std::unordered_map<uint64_t, action_callback> callbacks;

    public:
        GPU(unsigned id,
            Scheduler* scheduler, 
            network::controller::WorkerConnection* worker,
            unsigned worker_id,
            unsigned gpu_id,
            unsigned pages);

        // Thread safe
        void add_strategies(Request request) {
            incoming_strategies.push(request);
        }

        // Thread safe
        void add_result(std::shared_ptr<workerapi::Result> &result) {
            incoming_results.push(result);
        }

        // Not thread-safe.
        void check_pending();

    private:
        void send_action(InferAction* action);
        void send_action(LoadWeightsAction* action);
        void send_action(EvictWeightsAction* action);
        bool try_load(uint64_t available);
        std::vector<EvictWeightsAction*> evict_pages(unsigned required_pages);

        void handle_result(std::shared_ptr<workerapi::Result> &result);
        void infer_error(InferAction* action, std::shared_ptr<workerapi::ErrorResult> &error);
        void infer_success(InferAction* action, std::shared_ptr<workerapi::InferResult> &result);
        void infer_result(InferAction* action, std::shared_ptr<workerapi::Result> &result);
        void load_error(LoadWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &error);
        void load_success(LoadWeightsAction* action, std::shared_ptr<workerapi::LoadWeightsResult> &result);
        void load_result(LoadWeightsAction* action, std::shared_ptr<workerapi::Result> &result);
        void evict_error(EvictWeightsAction* action, std::shared_ptr<workerapi::ErrorResult> &result);
        void evict_success(EvictWeightsAction* action, std::shared_ptr<workerapi::EvictWeightsResult> &result);
        void evict_result(EvictWeightsAction* action, std::shared_ptr<workerapi::Result> &result);
    };
 public:

    // Thread-safe clockwork state
    LoadTracker* tracker;

    // Non-mutable so thread-safe
    std::vector<GPU*> gpus;
    std::vector<Model*> models;

 private:
    // Threads
    std::string actions_filename;
    ControllerActionTelemetryLogger* printer;
    std::thread network_printer;
    std::vector<std::thread> admission_threads;
    std::vector<std::thread> gpu_threads;

    // Messages
    tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
    tbb::concurrent_queue<Request> request_queue;

    // Actions
    std::atomic_flag actions_in_use;
    std::unordered_map<uint64_t, GPU*> outstanding_actions;

    // Diagnostic
    std::atomic_flag has_logged_inputs_status;


 public:

    // Called by GPU threads to register an action
    void add_action(uint64_t action_id, GPU* gpu);

    // Called when model loading has completed
    virtual void start(std::vector<network::controller::WorkerConnection*> workers,
                        ClockworkState &state);

    // The actual scheduler interface implementation, invoked by client network thread
    virtual void clientInfer(clientapi::InferenceRequest &request, 
        std::function<void(clientapi::InferenceResponse&)> callback);

    // The actual scheduler interface implementation, invoked by worker network thread
    virtual void resultFromWorker(std::shared_ptr<workerapi::Result> result);

 private:

    // Initialization methods
    void validate_clockwork_state(ClockworkState &state);
    void initialize_models(ClockworkState &state);
    void initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                    ClockworkState &state);
    void initialize_model_instances();
    void print_status();

    // The main thread run methods
    void run_admission_thread();
    void run_gpu_thread(std::vector<GPU*> gpus);

    // Logic of the dispatcher thread
    void handle_result(std::shared_ptr<workerapi::Result> &result);
    void handle_requests(std::vector<Request> &requests);
};

}
}
}

#endif // SRC_CLOCKWORK_CONTROLLER_CONCURRENT_INFER_AND_LOAD_SCHEDULER_H_