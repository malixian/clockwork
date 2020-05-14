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
    class Request {
     public:
        uint64_t id;
        clientapi::InferenceRequest request;
        clientapi::InferenceResponse response;
        std::function<void(clientapi::InferenceResponse&)> callback;
        uint64_t deadline;

        Request(clientapi::InferenceRequest request,
            std::function<void(clientapi::InferenceResponse&)> callback);
        ~Request();

        void set_result(char* output, size_t output_size);
        void set_error(int status, std::string message);
        void complete();
    };

    class Action {
     public:
        ControllerActionTelemetry telemetry;
        std::shared_ptr<workerapi::Infer> action = std::make_shared<workerapi::Infer>();
        std::shared_ptr<workerapi::ErrorResult> error = nullptr;
        std::shared_ptr<workerapi::InferResult> result = nullptr;
        std::vector<Request*> requests;

        explicit Action(unsigned model_id);
        ~Action();

        void batch();
        void unbatch();
        void set_error(std::shared_ptr<workerapi::ErrorResult> &error);
        void set_result(std::shared_ptr<workerapi::InferResult> &result);
        void complete();
    };

    class GPU;
    class Model {
     public:
        uint64_t request_id_seed = 0;
        unsigned id;

        GPU* assigned_gpu = nullptr;
        std::queue<Request*> queue;

        Model(unsigned id);

        void enqueue(Request* request);
        void check_timeouts();
        Action* try_dequeue(uint64_t expected_request_id);
    };

    class GPU {
     public:
        InferOnlyScheduler* scheduler;
        network::controller::WorkerConnection* worker;

        unsigned gpu_id;
        unsigned worker_id;
        unsigned outstanding = 0;

        void send_action(Action* action);
        void check_pending();
        void handle_error(Action* action, std::shared_ptr<workerapi::ErrorResult> &error);
        void handle_success(Action* action, std::shared_ptr<workerapi::InferResult> &result);
        void handle_result(Action* action, std::shared_ptr<workerapi::Result> &result);
        unsigned load_model_weights();
    };

    static const uint64_t print_interval = 10000000000UL; // 10 seconds
    static const uint64_t slo = 100000000UL; // 100ms
    static const bool print_debug = false;
    static const unsigned max_outstanding = 2;

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