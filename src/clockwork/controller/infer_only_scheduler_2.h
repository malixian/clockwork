// Copyright 2020 Max Planck Institute for Software Systems

#ifndef SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_2_H_
#define SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_2_H_

#include <atomic>
#include <algorithm>
#include <string>
#include <sstream>
#include "clockwork/controller/scheduler.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include "clockwork/thread.h"

namespace clockwork {
namespace scheduler {
namespace infer2 {

class InferOnlyScheduler : public Scheduler {
 public:

    static const uint64_t print_interval = 10000000000UL;
    static const bool print_debug = false;


    static const uint64_t slo = 100000000UL; // 100ms SLO
    static const uint64_t buffer = 5000000UL; // Aim for an SLO this much prior to actual SLO
    static const uint64_t default_clock = 1380; // default gpu clock speed
    static const int estimate_window_size = 10; // Estimate execution time using last 10 measurements
    static const float estimate_percentile; // Percentile to use for estimation; 0.99 (effectively max)
    static const uint64_t latest_delta = 3000000UL; // Actions can run up to 3ms behind schedule before the worker will drop them
    static const uint64_t schedule_ahead = 10000000UL; // schedule 10ms into the future
    static const uint64_t max_allowable_exec_time = 18000000UL; // for batching, never consider batch sizes that exceed 18ms exec time (too big)



    class Request {
     public:
        uint64_t id;
        clientapi::InferenceRequest request;
        clientapi::InferenceResponse response;
        std::function<void(clientapi::InferenceResponse&)> callback;
        uint64_t deadline;
        uint64_t departure;


        Request(clientapi::InferenceRequest request,
            std::function<void(clientapi::InferenceResponse&)> callback);
        ~Request();

        void set_result(char* output, size_t output_size);
        void set_error(int status, std::string message);

        // Returns true if the result was successful and within the deadline
        void timeout();
        bool complete(uint64_t now);
    };

    class Model;
    class Action {
     public:
        Model* model;
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::Infer> action = std::make_shared<workerapi::Infer>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::InferResult> result = nullptr;
        std::vector<Request*> requests;

        explicit Action(Model* model);
        ~Action();

        void batch();
        void unbatch();
        void set_expectations(uint64_t exec_start, uint64_t duration, int clock);
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::InferResult> &result);

        // Returns the fraction of successful requests
        float complete(uint64_t now);
    };

    class InferStrategy;
    class GPU;
    class Model {
     private:
        std::vector<unsigned> supported_batch_sizes;
        std::vector<unsigned> batch_lookup_;
        unsigned max_batch_size;

        std::map<unsigned, uint64_t> estimates;
        std::map<unsigned, util::SlidingWindow*> estimators;
        uint64_t request_id_seed = 0;

        std::queue<Request*> queue;

     public:

        unsigned id;

        Model(BatchedModelState &state);

        std::vector<InferStrategy*> enqueue(Request* request);

        // For num_requests requests, what is the maximum batch size we could execute?
        unsigned batch_lookup(unsigned num_requests);
        
        void check_timeouts(GPU* gpu, uint64_t free_at);
        Action* try_dequeue(GPU* gpu, uint64_t gpu_free_at, InferStrategy* strategy);
        void add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock);
        uint64_t estimate(unsigned batch_size);
        uint64_t estimate(unsigned batch_size, int clock);
    };

    class GPU {
     public:
        InferOnlyScheduler* scheduler;
        network::controller::WorkerConnection* worker;
        util::WorkerTracker tracker;

        unsigned gpu_id;
        unsigned worker_id;

        GPU();

        void send_action(Action* action);
        void check_pending();
        void handle_error(Action* action, std::shared_ptr<workerapi::ErrorResult> &error);
        void handle_success(Action* action, std::shared_ptr<workerapi::InferResult> &result);
        void handle_result(Action* action, std::shared_ptr<workerapi::Result> &result);
        unsigned load_model_weights();
    };

    class InferStrategy {
    public:
        uint64_t priority;
        uint64_t deadline;
        uint64_t request_id;
        unsigned batch_size;
        Model* model;

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

    // Clockwork State
    std::vector<GPU*> gpus;
    std::map<unsigned, Model*> models;

    // Threads
    std::string actions_filename;
    ControllerActionTelemetryLogger* printer;
    std::thread thread;

    // Messages
    tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
    tbb::concurrent_queue<Request*> request_queue;

    std::priority_queue<InferStrategy*, std::deque<InferStrategy*>, InferStrategy::Comparator> queue;

    struct OutstandingAction { GPU* gpu; Action* action; };
    std::unordered_map<uint64_t, OutstandingAction> outstanding_actions;


    InferOnlyScheduler(std::string actions_filename = "/local/clockwork_action_log.tsv");


    void validate_clockwork_state(ClockworkState &state);
    void initialize_models(ClockworkState &state);
    void initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                    ClockworkState &state);


    // Called when model loading has completed
    virtual void start(std::vector<network::controller::WorkerConnection*> workers,
                        ClockworkState &state);
    void run();

    // The actual scheduler interface implementation, invoked by worker network thread
    virtual void resultFromWorker(std::shared_ptr<workerapi::Result> result);
    void handle_result(std::shared_ptr<workerapi::Result> result);

    // The actual scheduler interface implementation, invoked by client network thread
    virtual void clientInfer(clientapi::InferenceRequest &request, 
        std::function<void(clientapi::InferenceResponse&)> callback);
    void handle_request(Request* request);
};

}
}
}

#endif // SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_2_H_