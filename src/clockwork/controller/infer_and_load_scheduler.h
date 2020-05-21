// Copyright 2020 Max Planck Institute for Software Systems

#ifndef SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_3_H_
#define SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_3_H_

#include <atomic>
#include <algorithm>
#include <string>
#include <sstream>
#include "clockwork/controller/scheduler.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include "clockwork/thread.h"
#include "clockwork/api/worker_api.h"

namespace clockwork {
namespace scheduler {
namespace infer3 {

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



    class WorkTracker {
     public:
        struct Demand {
            int model;
            int64_t work;
            int64_t deficit;
            int gpu; // -1 if no gpu
        };

     private:
        static const int64_t capacity = Scheduler::slo; // For now just use the slo
        struct Model {
            int id;
            int64_t count = 0;
            int64_t work = 0;
            int64_t deficit = 0;
            int64_t surplus = 0;
            int gpu_count = 0;
            std::vector<bool> gpus;
            std::vector<bool> loading;
            uint64_t seqno = 0;
        };
        struct CompareModelDeficit {
            bool operator() (Model* a, Model* b) {
                if (a->count == 0) {
                    return b->count == 0 ? a->seqno > b->seqno : false;
                } else {
                    return b->count == 0 ? true : a->deficit > b->deficit;
                }
            }
        } sort_by_deficit;

        struct GPU {
            int id;
            int64_t work = 0;
            int model_count = 0;
            std::vector<bool> models;
        };

        uint64_t seqno_seed = 0;

        std::vector<Model> models;
        std::vector<Model*> ordered;
        std::vector<GPU> gpus;

     public:

        WorkTracker(int num_gpus, int num_models);

        Demand addRequest(int model_id, int64_t size, uint64_t slo);
        void requestCompleted(Demand &demand);
        int loadModel(int gpu_id, bool requires_eviction=false);
        void loadModelComplete(int gpu_id, int model_id, bool success);
        int evictModel(int gpu_id);
    };


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

        void updatePriority(Model &model) {
            // Include all work for evict priority, but only outstanding work for load priority
            bool is_empty = (model.outstanding + model.completed) == 0;
            for (unsigned i = 0; i < n_gpus; i++) {
                if (model.gpus[i]) {
                    model.priorities[i]->priority = model.outstanding + model.completed;
                } else {
                    model.priorities[i]->priority = model.outstanding;                  
                }
                model.priorities[i]->is_empty = is_empty;
            }
            for (unsigned i = 0; i < n_gpus; i++) {
                if (model.gpus[i]) {
                    int64_t serving_capacity = (capacity * model.allocations[i]) / gpus[i].outstanding;
                    auto &preference = model.priorities[i]->preference;
                    for (unsigned j = 0; j < n_gpus; j++) {
                        if (i != j) {
                            model.priorities[j]->priority -= serving_capacity;
                        }
                        // if (!model.gpus[j] || model.priorities[j]->preference > preference) {
                        //     model.priorities[j]->priority -= serving_capacity;
                        // }
                    }
                }
            }
        }

        void clearWork(Model &model) {
            for (unsigned i = 0; i < n_gpus; i++) {
                gpus[i].outstanding -= model.allocations[i];
                model.allocations[i] = 0;
            }
        }

        void distributeWork(Model &model) {
            if (model.gpu_count == 0) return;

            clearWork(model);

            double total_weight = 0;
            std::vector<double> weights(n_gpus, 0);
            for (unsigned i = 0; i < n_gpus; i++) {
                if (model.gpus[i]) {
                    weights[i] = capacity / ((double) gpus[i].outstanding);
                    total_weight += weights[i];
                }
            }

            for (unsigned i = 0; i < n_gpus; i++) {
                if (model.gpus[i]) {
                    auto allocation = (model.outstanding + model.completed) * (weights[i] / total_weight);
                    model.allocations[i] = allocation;
                    gpus[i].outstanding += allocation;
                }
            }
        }

        void addGPU(Model &model, GPU &gpu) {
            model.gpus[gpu.id] = true;
            model.loading[gpu.id] = true;
            gpu.models[model.id] = true;

            model.priorities[gpu.id]->preference = model.gpu_count++;
            distributeWork(model);
            updatePriority(model);
        }

        void removeGPU(Model &model, GPU &gpu) {
            model.gpus[gpu.id] = false;
            model.loading[gpu.id] = false;
            gpu.models[model.id] = false;

            if (--model.gpu_count == 0) {
                clearWork(model);
                model.seqno = seqno_seed++;
            } else {
                distributeWork(model);

                // Decrement preferences
                int preference = model.priorities[gpu.id]->preference;
                for (unsigned i = 0; i < n_gpus; i++) {
                    if (model.gpus[i] && model.priorities[i]->preference > preference) {
                        model.priorities[i]->preference--;
                    }
                }
            }
            updatePriority(model);
        }

        void checkRequests() {
            uint64_t now = util::now();
            while (!requests.empty() && requests.top().time < now) {
                auto &request = requests.top();
                auto &model = models[request.model_id];
                model.completed -= request.size;

                distributeWork(model);
                updatePriority(model);
                requests.pop();
            }
        }

     public:

        WorkTracker2(int num_gpus, int num_models) : n_models(num_models), n_gpus(num_gpus) {
            gpus.resize(num_gpus);
            for (unsigned i = 0; i < num_gpus; i++) {
                gpus[i].id = i;
                gpus[i].models.resize(num_models, false);
                gpus[i].modelorder.reserve(num_models);
            }

            models.resize(num_models);
            for (unsigned i = 0; i < num_models; i++) {
                auto &model = models[i];
                model.id = i;
                model.gpus.resize(num_gpus, false);
                model.loading.resize(num_gpus, false);
                model.allocations.resize(num_gpus, 0);

                model.priorities.resize(num_gpus);
                for (unsigned j = 0; j < num_gpus; j++) {
                    auto priority = new ModelPriority(&model);
                    model.priorities[j] = priority;
                    gpus[j].modelorder.push_back(priority);
                }
            }            
        }

        Demand addRequest(int model_id, int64_t size, uint64_t slo) {
            // Complete any pending requests
            checkRequests();

            // First, normalize the work to the SLO
            size = (size * capacity) / slo;

            // Create the demand
            Scheduler::WorkTracker2::Demand demand;
            demand.size = size;
            demand.model_id = model_id;

            Scheduler::WorkTracker2::Request request;
            request.size = size;
            request.model_id = model_id;
            request.time = util::now() + slo;
            requests.push(request);

            // Get the model
            auto &model = models[model_id];
            model.outstanding += size;
            
            // Update the model's priorities
            distributeWork(model);
            updatePriority(model);

            return demand;
        }

        void requestCompleted(Demand &demand) {
            auto &model = models[demand.model_id];
            model.outstanding -= demand.size;
            model.completed += demand.size;

            distributeWork(model);
            updatePriority(model);
        }

        int loadModel(int gpu_id, bool requires_eviction = false) {
            // Complete any pending requests
            checkRequests();

            auto &gpu = gpus[gpu_id];

            std::sort(gpu.modelorder.begin(), gpu.modelorder.end(), sort_by_priority);

            unsigned seen = 0;
            for (auto &priority : gpu.modelorder) {
                if (requires_eviction && seen == gpu.model_count) break;

                if (priority->priority < 0) break; // all demands satisfied

                Model &model = *priority->model;
                if (!model.gpus[gpu_id] && !model.loading[gpu_id]) {
                    addGPU(model, gpu);
                    gpu.model_count++;
                    return model.id;
                }

                if (model.gpus[gpu_id]) {
                    seen++;
                }
            }

            return -1; // all models loaded on all gpus
        }

        void loadModelComplete(int gpu_id, int model_id, bool success) {
            // Complete any pending requests
            checkRequests();

            auto &model = models[model_id];
            auto &gpu = gpus[gpu_id];

            model.loading[gpu_id] = false;

            if (!success) {
                removeGPU(model, gpu);
            }
        }

        int evictModel(int gpu_id) {
            // Complete any pending requests
            checkRequests();

            auto &gpu = gpus[gpu_id];

            std::sort(gpu.modelorder.begin(), gpu.modelorder.end(), sort_by_priority);

            for (int i = n_models - 1; i >= 0; i--) {
                auto &priority = gpu.modelorder[i];
                Model &model = *priority->model;
                if (model.gpus[gpu_id] && !model.loading[gpu_id]) {
                    removeGPU(model, gpus[gpu_id]);
                    gpu.model_count--;
                    return model.id;
                }
            }

            return -1; // all models are unloaded on all gpus

        }
    };


    class Model;
    class RequestImpl {
     public:
        bool has_completed = false;
        uint64_t id;
        Model* model = nullptr;
        clientapi::InferenceRequest request;
        clientapi::InferenceResponse response;
        std::function<void(clientapi::InferenceResponse&)> callback;
        uint64_t deadline;
        uint64_t departure;

        WorkTracker2::Demand demand;


        RequestImpl(clientapi::InferenceRequest request,
            std::function<void(clientapi::InferenceResponse&)> callback);
        ~RequestImpl();

        void set_result(char* output, size_t output_size);
        void set_error(int status, std::string message);

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
        unsigned version = 0;
        ModelInstance(GPU* gpu, Model* model): gpu(gpu), model(model) {}
    };

    class Model {
     private:
        std::vector<unsigned> supported_batch_sizes;
        std::vector<unsigned> batch_lookup_;
        unsigned max_batch_size;

        std::map<unsigned, uint64_t> estimates;
        std::map<unsigned, util::SlidingWindow*> estimators;
        util::SlidingWindow* weights_estimator;
        uint64_t weights_estimate;
        uint64_t request_id_seed = 0;

        std::deque<Request> queue;

     public:
        Scheduler* scheduler;
        std::vector<ModelInstance*> instances; // an instance of a model on a GPU
        unsigned num_weights_pages;

        unsigned id;

        Model(BatchedModelState &state);

        // Enqueues the request to this model, then enqueues InferStrategies to all active ModelInstances
        void enqueue(Request request);

        // Enqueues InferStrategies for all requests to this ModelInstance
        void enqueue(ModelInstance* instance);

        // For num_requests requests, what is the maximum batch size we could execute?
        unsigned batch_lookup(unsigned num_requests);

        void check_timeouts(uint64_t free_at);
        InferAction* try_dequeue(GPU* gpu, uint64_t gpu_free_at, InferStrategy* strategy);
        void add_measurement(unsigned batch_size, uint64_t duration, unsigned gpu_clock);
        void add_weights_measurement(uint64_t duration);
        uint64_t estimate(unsigned batch_size);
        uint64_t estimate(unsigned batch_size, int clock);
        uint64_t estimate_weights();
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
        network::controller::WorkerConnection* worker;
        Scheduler* scheduler;
        util::WorkerTracker exec;
        util::WorkerTracker loadweights;

        unsigned free_pages;
        bool eviction_required = false;

        std::vector<ModelInstance*> instances;

        unsigned id; // a unique id for this (worker_id, gpu_id)
        unsigned worker_id; // the id of the worker
        unsigned gpu_id; // the id of the gpu on the worker

        std::queue<Loading> loading;
        std::priority_queue<InferStrategy*, std::deque<InferStrategy*>, InferStrategy::Comparator> queue;

        GPU();


        void send_action(InferAction* action);
        void send_action(LoadWeightsAction* action);
        void send_action(EvictWeightsAction* action);
        void check_pending();
        void evict_pages(unsigned required_pages);
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

    // Clockwork State
    WorkTracker2* tracker;
    std::vector<GPU*> gpus;
    std::vector<Model*> models;
    std::queue<Request> requests;

    // Threads
    std::string actions_filename;
    ControllerActionTelemetryLogger* printer;
    std::thread thread;

    // Messages
    tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> result_queue;
    tbb::concurrent_queue<Request> request_queue;

    struct OutstandingAction { int type; GPU* gpu; void* action; };
    std::unordered_map<uint64_t, std::function<void(std::shared_ptr<workerapi::Result>&)>> outstanding_actions;


    Scheduler(std::string actions_filename = "/local/clockwork_action_log.tsv");


    void validate_clockwork_state(ClockworkState &state);
    void initialize_models(ClockworkState &state);
    void initialize_gpus(std::vector<network::controller::WorkerConnection*> workers,
                    ClockworkState &state);
    void initialize_model_instances();
    void print_status();


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
    void handle_request(Request request);
};

}
}
}

#endif // SRC_CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_3_H_