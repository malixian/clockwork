// Copyright 2020 Max Planck Institute for Software Systems

#ifndef SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_H_
#define SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_H_

#include <atomic>
#include <algorithm>
#include "clockwork/controller/scheduler.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include "clockwork/thread.h"


namespace clockwork {

class InferOnlyScheduler : public Scheduler {
 public:

    static const uint64_t print_interval = 10000000000UL; // 10 seconds
    static const uint64_t slo = 100000000UL; // 100ms
    static const uint64_t buffer = 2000000UL; // 2ms buffer
    static const uint64_t default_clock = 1380; // 2ms buffer
    static const bool print_debug = false;
    static const int estimate_window_size = 100;
    static const float estimate_percentile;

    static const uint64_t schedule_ahead = 4000000UL; // schedule 4ms into the future



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
        void timeout(uint64_t now);
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
        uint64_t expected_duration;
        uint64_t expected_exec_complete;
        int expected_gpu_clock;

        explicit Action(Model* model);
        ~Action();

        void batch();
        void unbatch();
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::InferResult> &result);

        // Returns the fraction of successful requests
        float complete(uint64_t now);
    };

    class GPU;
    class Model {
     public:
        std::map<unsigned, uint64_t> estimates;
        std::map<unsigned, util::SlidingWindow*> estimators;

        uint64_t request_id_seed = 0;
        unsigned id;

        GPU* assigned_gpu = nullptr;
        std::queue<Request*> queue;

        Model(BatchedModelState &state);

        void enqueue(Request* request);
        void check_timeouts(uint64_t now);
        Action* try_dequeue(uint64_t gpu_free_at, uint64_t expected_request_id);
        void add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock);
        uint64_t estimate(unsigned batch_size);
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

    // Clockwork State
    std::vector<GPU*> gpus;
    std::map<unsigned, Model*> models;

    // Threads
    std::string actions_filename;
    ControllerActionTelemetryLogger* printer;
    std::thread thread;

    // Algorithm State
    std::queue<GPU*> gpu_fifo;

    // Messages
    tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
    tbb::concurrent_queue<Request*> request_queue;

    struct QueueElement { uint64_t request_id; Model* model; };
    std::queue<QueueElement> queue;

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

#endif // SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_H_