// Copyright 2020 Max Planck Institute for Software Systems

#ifndef SRC_CLOCKWORK_CONTROLLER_CONCURRENT_INFER_AND_LOAD_SCHEDULER_H_
#define SRC_CLOCKWORK_CONTROLLER_CONCURRENT_INFER_AND_LOAD_SCHEDULER_H_

#include <atomic>
#include <algorithm>
#include <string>
#include <sstream>
#include "clockwork/controller/scheduler.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include "clockwork/thread.h"
#include "clockwork/api/worker_api.h"
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


    static const uint64_t slo = 100000000UL; // 100ms SLO
    static const uint64_t buffer = 5000000UL; // Aim for an SLO this much prior to actual SLO
    static const uint64_t default_clock = 1380; // default gpu clock speed
    static const int estimate_window_size = 10; // Estimate execution time using last 10 measurements
    static const float estimate_percentile; // Percentile to use for estimation; 0.99 (effectively max)
    static const uint64_t latest_delta = 3000000UL; // Actions can run up to 3ms behind schedule before the worker will drop them
    static const uint64_t schedule_ahead = 10000000UL; // schedule 10ms into the future
    static const uint64_t max_allowable_exec_time = 18000000UL; // for batching, never consider batch sizes that exceed 18ms exec time (too big)



    class WorkTracker2 {
     public:
        struct Demand {
            int model_id;
            int64_t size;
        };

     private:
        static const int64_t capacity = Scheduler::slo; // For now just use the slo
        struct ModelPriority;
        struct Model {
            int id;
            int gpu_count = 0;
            std::vector<bool> gpus;
            std::vector<bool> loading;
            int64_t outstanding = 0;
            int64_t completed = 0;
            std::vector<uint64_t> allocations;
            std::vector<ModelPriority*> priorities;
            uint64_t seqno = 0;
        };

        struct ModelPriority {
            int64_t priority = 0;
            bool is_empty = true;
            int preference = 0;
            Model* model;
            ModelPriority(Model* model) : model(model) {}
        };

        struct CompareModelPriority {
            bool operator() (ModelPriority* a, ModelPriority* b) {
                if (a->is_empty) {
                    if (b->is_empty) {
                        return a->model->seqno > b->model->seqno;
                    } else {
                        return false;
                    }
                } else {
                    if (b->is_empty) {
                        return true;
                    } else {
                        return a->priority > b->priority;
                    }
                }
            }
        } sort_by_priority;

        struct GPU {
            int id;
            int64_t outstanding = 1000000UL; // always assume 1ms outstanding work
            unsigned model_count = 0;
            std::vector<bool> models;
            std::vector<ModelPriority*> modelorder;
        };

        struct Request {
            int model_id;
            int64_t size;
            uint64_t time;

            friend bool operator < (const Request& lhs, const Request &rhs) {
                return lhs.time < rhs.time;
            }
            friend bool operator > (const Request& lhs, const Request &rhs) {
                return lhs.time > rhs.time;
            }
        };

        uint64_t seqno_seed = 0;
        std::vector<Model> models;
        std::vector<GPU> gpus;
        const unsigned n_models;
        const unsigned n_gpus;

        std::priority_queue<Request, std::vector<Request>, std::greater<Request>> requests;

        void updatePriority(Model &model);
        void clearWork(Model &model);
        void distributeWork(Model &model);
        void addGPU(Model &model, GPU &gpu);
        void removeGPU(Model &model, GPU &gpu);
        void checkRequests();

     public:
        tbb::queuing_mutex mutex;

        WorkTracker2(int num_gpus, int num_models);
        Demand addRequest(int model_id, int64_t size, uint64_t slo);
        void requestCompleted(Demand &demand);
        int loadModel(int gpu_id, bool requires_eviction = false);
        void loadModelComplete(int gpu_id, int model_id, bool success);
        int evictModel(int gpu_id);
    };


    class Model;
    class RequestImpl {
     public:
        uint64_t id;
        uint64_t slo;
        uint64_t exec_slo;
        uint64_t weights_slo;
        uint64_t deadline;
        Model* model = nullptr;
        clientapi::InferenceRequest request;
        clientapi::InferenceResponse response;

        WorkTracker2::Demand demand;

     private:
        std::atomic_bool locked;

        std::function<void(clientapi::InferenceResponse&)> callback;

     public:
        RequestImpl(clientapi::InferenceRequest request,
            std::function<void(clientapi::InferenceResponse&)> callback);
        ~RequestImpl();

        void set_model(Model* model);
        void set_slo(uint64_t default_slo);
        void set_result(char* output, size_t output_size);
        void set_error(int status, std::string message);

        void lock();

        // Returns true if the result was successful and within the deadline
        void timeout();
        bool complete(uint64_t now);
        void finalize();
    };
    typedef std::shared_ptr<RequestImpl> Request;

    class InferAction {
     public:
        Model* model;
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::Infer> action = std::make_shared<workerapi::Infer>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::InferResult> result = nullptr;
        std::vector<Request> requests;

        explicit InferAction(Model* model);
        ~InferAction();

        void batch();
        void unbatch();
        void set_expectations(uint64_t exec_start, uint64_t duration, int clock);
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::InferResult> &result);

        // Returns the fraction of successful requests
        float complete(uint64_t now);
    };

    class ModelInstance;
    class LoadWeightsAction {
     public:
        ModelInstance* instance;
        unsigned version;
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::LoadWeights> action = std::make_shared<workerapi::LoadWeights>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::LoadWeightsResult> result = nullptr;

        explicit LoadWeightsAction(ModelInstance* instance);

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

    class InferStrategy;
    class GPU;

    class ModelInstance {
     public:
        GPU* gpu = nullptr;
        Model* model = nullptr;
        bool loaded = false;
        bool loading = false;
        unsigned version = 0;
        ModelInstance(GPU* gpu, Model* model): gpu(gpu), model(model) {}
    };

    class Model {
     public:
        unsigned id;
        Scheduler* scheduler;
        unsigned num_weights_pages;
        std::vector<ModelInstance*> instances;
        uint64_t b1_exec;
        std::atomic_int copies_loaded = 0;
        std::atomic_int requests_queued = 0;

     private:
        tbb::queuing_mutex mutex;

        std::vector<unsigned> supported_batch_sizes;
        std::vector<unsigned> batch_lookup_;
        unsigned max_batch_size;

        std::map<unsigned, uint64_t> estimates;
        std::map<unsigned, util::SlidingWindow*> estimators;

        std::atomic_flag weights_in_use;
        util::SlidingWindow* weights_estimator;
        uint64_t weights_estimate;

        uint64_t request_id_seed = 0;

        std::deque<Request> queue;

     public:

        Model(BatchedModelState &state);

        uint64_t estimate_weights();

        // Enqueues the request to this model, then enqueues InferStrategies to all active ModelInstances
        void enqueue(Request request);

        // Enqueues InferStrategies for all requests to this ModelInstance
        void enqueue(ModelInstance* instance);

        // Gets actions to execute for this model
        InferAction* try_dequeue(uint64_t gpu_free_at, unsigned gpu_clock, InferStrategy* strategy, bool &retry);

        // GPUs can add new measurements
        void add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock);
        void add_weights_measurement(uint64_t duration);

     private:

        // For num_requests requests, what is the maximum batch size we could execute?
        unsigned batch_lookup(unsigned num_requests);

        void check_timeouts(uint64_t free_at);
        uint64_t estimate(unsigned batch_size);
        uint64_t estimate(unsigned batch_size, int clock);
    };

    class InferStrategy {
    public:
        uint64_t priority;
        uint64_t deadline;
        uint64_t request_id;
        unsigned batch_size;
        unsigned version;
        ModelInstance* instance;

        struct Comparator {
            bool operator()(const InferStrategy* lhs, const InferStrategy* rhs) {
                return lhs->priority > rhs->priority;
            }
        };

        std::string str() {
            std::stringstream ss;
            ss << "S p=" << priority << " d=" << deadline << " rid=" << request_id << " b=" << batch_size;
            return ss.str();
        }
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
        util::WorkerTracker exec;
        util::WorkerTracker loadweights;

        unsigned free_pages;
        bool eviction_required = false;
        uint64_t last_load = 0;
        uint64_t last_exec = 0;

        std::queue<Loading> loading;
        std::priority_queue<InferStrategy*, std::deque<InferStrategy*>, InferStrategy::Comparator> queue;

        // Strategies enqueued by models
        tbb::concurrent_queue<InferStrategy*> strategies;

        // Results enqueued by network
        tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> results;

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
        void add_strategy(InferStrategy* strategy) {
            strategies.push(strategy);
        }

        // Thread safe
        void add_result(std::shared_ptr<workerapi::Result> result) {
            results.push(result);
        }

        // Not thread-safe.
        void check_pending();

    private:
        void send_action(InferAction* action);
        void send_action(LoadWeightsAction* action);
        void send_action(EvictWeightsAction* action);
        bool try_load(uint64_t available);
        void evict_pages(unsigned required_pages);

        void handle_result(std::shared_ptr<workerapi::Result> result);
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
    WorkTracker2* tracker;

    // Non-mutable so thread-safe
    std::vector<GPU*> gpus;
    std::vector<Model*> models;

    uint64_t default_slo;

 private:
    // All requests to time out
    std::queue<Request> requests;

    // Threads
    std::string actions_filename;
    ControllerActionTelemetryLogger* printer;
    std::thread thread;
    std::vector<std::thread> gpu_threads;

    // Messages
    tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
    tbb::concurrent_queue<Request> request_queue;

    // Actions
    std::atomic_flag actions_in_use;
    std::unordered_map<uint64_t, GPU*> outstanding_actions;

 public:

    // Called by GPU threads to register an action
    void add_action(uint64_t action_id, GPU* gpu);


    Scheduler(uint64_t default_slo = 100000000UL, std::string actions_filename = "/local/clockwork_action_log.tsv");


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
    void run();
    void run_gpu_thread(std::vector<GPU*> gpus);

    // Logic of the dispatcher thread
    void handle_result(std::shared_ptr<workerapi::Result> result);
    void handle_request(Request request);
};

}
}
}

#endif // SRC_CLOCKWORK_CONTROLLER_CONCURRENT_INFER_AND_LOAD_SCHEDULER_H_