#ifndef _CLOCKWORK_SMART_SCHEDULER_H_
#define _CLOCKWORK_SMART_SCHEDULER_H_

#include <string>
#include "clockwork/telemetry/controller_action_logger.h"
#include "scheduler.h"

namespace clockwork {

#define SCHEDULING_EPOCH (3000000ULL)     // 3ms
#define SCHEDULE_AHEAD (20000000ULL)      // 10ms
#define NETWORK_TRANSFER_DELAY 1000000ULL  // 1ms
#define PCI_AVERAGE_SLACK 1000000ULL      // 1ms
#define INFER_SLACK 500000ULL             // 0.5ms
#define WEIGHTS_EVICT_LATENCY 1000000ULL  // 1ms
#define REPLICATION_SENSITIVITY \
  50  // it means we must be observing at least 50 drops to trigger replication
#define CLK_FREQ_DEFAULT 1380  // MHz

// ------ EXECUTION PROFILERS

class SmartExecutionProfiler {
 public:
  float percentile;
  unsigned window_size;
  unsigned clk_freq;

  std::vector<unsigned> batch_sizes;
  std::map<unsigned, uint64_t> estimates;  // store latency * freq
  std::map<unsigned, util::SlidingWindow>
      sliding_windows;  // track latency * freq

  void set_batch_sizes(std::vector<unsigned> &sizes);
  void set_estimates(std::map<unsigned, uint64_t> latencies);
  void set_window_size(unsigned size);

  uint64_t get_latency_estimate(unsigned batch_size);
  unsigned get_max_batch_size(uint64_t slack, unsigned limit);

  void insert(unsigned batch, uint64_t latency, unsigned freq);

  void update_estimate(unsigned batch);
  void update_all_estimates();

  SmartExecutionProfiler()
      : percentile(99), window_size(100), clk_freq(CLK_FREQ_DEFAULT) {}
};

class SmartWeightsProfiler {
 public:
  uint64_t estimate;  // store the estimate in ns
  uint64_t get_estimate() { return estimate; }
};

void SmartExecutionProfiler::set_batch_sizes(std::vector<unsigned> &sizes) {
  this->batch_sizes = sizes;
}

void SmartExecutionProfiler::set_estimates(
    std::map<unsigned, uint64_t> latencies) {
  estimates.clear();
  for (auto const &l : latencies) {
    estimates.insert(std::make_pair(l.first, l.second * CLK_FREQ_DEFAULT));
  }
}

void SmartExecutionProfiler::set_window_size(unsigned size) {
  this->window_size = size;
  for (auto const &s : batch_sizes) {
    sliding_windows[s] = util::SlidingWindow(size);
  }
}

uint64_t SmartExecutionProfiler::get_latency_estimate(unsigned batch_size) {
  return estimates[batch_size] / clk_freq;
}

unsigned SmartExecutionProfiler::get_max_batch_size(uint64_t slack,
                                                    unsigned limit) {
  unsigned current_max = 1;
  for (auto const &s : batch_sizes) {
    if (s > limit) {
      break;
    }
    if (get_latency_estimate(s) > slack) {
      break;
    }
    current_max = s;
  }
  return current_max;
}

void SmartExecutionProfiler::insert(unsigned batch, uint64_t latency,
                                    unsigned freq) {
  clk_freq = freq;
  auto it = sliding_windows.find(batch);
  it->second.insert(latency * freq);
  update_estimate(batch);
}

/* Assumptions:
   -- sliding_windows.find(batch) != sliding_windows.end()
   -- estimates.find(batch) != estimates.end() */
void SmartExecutionProfiler::update_estimate(unsigned batch) {
  util::SlidingWindow &sliding_window = sliding_windows[batch];

  if (sliding_window.get_size() == 0) {
    return;
  }

  unsigned rank = sliding_window.get_size() - 1;  // by default, get max value

  if (sliding_window.get_size() >= window_size) {
    rank = ceil(window_size * (percentile / 100.0)) - 1;
  }

  estimates[batch] = sliding_window.get_value(rank);
}

void SmartExecutionProfiler::update_all_estimates() {
  for (auto const &s : batch_sizes) {
    update_estimate(s);
  }
}

// ------

class SmartGPU {
 public:
  unsigned id;
  unsigned worker_id;
  unsigned gpu_index;
  uint64_t gpu_idle_at;
  uint64_t pci_idle_at;
  uint64_t available_pages;
  uint64_t total_pages;
  std::set<unsigned> loaded_models;
  std::vector<unsigned> lru_loaded_models;

  SmartGPU(unsigned id, unsigned worker_id, unsigned gpu_index,
           unsigned total_pages)
      : id(id),
        worker_id(worker_id),
        gpu_index(gpu_index),
        gpu_idle_at(0),
        pci_idle_at(0),
        total_pages(total_pages),
        available_pages(total_pages) {}

  void update_lru(unsigned model_id) {
    auto pos =
        std::find(lru_loaded_models.begin(), lru_loaded_models.end(), model_id);
    if (pos != std::end(lru_loaded_models)) {
      lru_loaded_models.erase(pos);
    }
    lru_loaded_models.insert(lru_loaded_models.begin(), model_id);
  }

  void evict_model(unsigned model_id) {
    auto pos =
        std::find(lru_loaded_models.begin(), lru_loaded_models.end(), model_id);
    if (pos != std::end(lru_loaded_models)) {
      lru_loaded_models.erase(pos);
      loaded_models.erase(model_id);
    }
  }
};

class SmartModel {
 public:
  unsigned id;
  std::map<unsigned, uint64_t> weights_available_at;  // gpu_id -> timestamp
  BatchedModelState state;
  SmartModel(unsigned id, BatchedModelState &state) : id(id), state(state) {}
};

class SmartInferRequest {
 public:
  uint64_t id;
  std::shared_ptr<clientapi::InferenceRequest> request_ptr;
  uint64_t arrived;
  uint64_t deadline;
  uint64_t earliest;
  uint64_t start_time;
  uint64_t finish_time;

  SmartInferRequest(uint64_t id, uint64_t arrived, uint64_t deadline,
                    std::shared_ptr<clientapi::InferenceRequest> request_ptr)
      : id(id), arrived(arrived), deadline(deadline), request_ptr(request_ptr) {
    earliest = util::now();
    start_time = 0;
    finish_time = 0;
  }
};

class SmartBatch {
  uint64_t id;
  unsigned model_id;
  unsigned batch_size;
  std::vector<std::shared_ptr<SmartInferRequest>> requests;
  uint64_t earliest;
  uint64_t deadline;
  uint64_t start_time;
  uint64_t finish_time;

  SmartBatch(uint64_t id, unsigned model_id,
             std::shared_ptr<SmartInferRequest> request)
      : id(id), batch_size(1), model_id(model_id) {
    earliest = request->earliest;
    deadline = request->deadline;
    start_time = request->start_time;
    finish_time = request->finish_time;
    requests.push_back(request);
  }

  void add_to_batch(std::shared_ptr<SmartInferRequest> request) {
    requests.push_back(request);
  }
};

class SmartScheduler : public Scheduler {
 public:
  // telemetry

  ControllerActionTelemetryLogger *logger = nullptr;
  std::map<unsigned, ControllerActionTelemetry *> action_telemetry_map;
  std::mutex mtx_telemetry;

  // Profilers

  std::map<unsigned, SmartExecutionProfiler>
      execution_profiler;  // model_id -> execution_profiler
  std::map<unsigned, SmartWeightsProfiler>
      weights_profiler;  // model_id -> weights_profiler
  std::mutex mtx_profiler;

  //

  std::atomic_uint64_t action_id_seed;
  std::atomic_uint64_t global_request_id;

  uint64_t slo;

  // workers and gpus
  std::vector<network::controller::WorkerConnection *> workers;
  std::map<unsigned, std::shared_ptr<SmartGPU>> gpu_map;
  std::vector<std::shared_ptr<SmartGPU>> gpu_list;

  // models
  std::map<unsigned, std::shared_ptr<SmartModel>> models;

  // mapping models to gpus
  std::map<unsigned, std::vector<unsigned>>
      model_gpu_map;  // model_id -> vector of gpu_ids

  // check if a model is hot or not
  std::set<unsigned>
      global_model_cache_stat;  // models that are cached, at least on one gpu

  // drop count of each model, at each scheduling phase
  std::map<unsigned, unsigned> model_drop_count;

  // global request_queue
  std::vector<std::shared_ptr<SmartInferRequest>> request_queue;
  std::mutex mtx_request_queue;

  std::map<unsigned, std::vector<std::shared_ptr<SmartInferRequest>>>
      gpu_request_queue;

  // action callbacks
  std::map<uint64_t, std::function<void(std::shared_ptr<workerapi::Result>)>>
      action_callbacks;
  std::mutex mtx_action_callbacks;

  std::map<uint64_t, std::function<void(clientapi::InferenceResponse &)>>
      inference_callbacks;
  std::mutex mtx_inference_callbacks;

  // this thread does the scheduling
  std::thread scheduler_thread;

  void gpu_local_schedule(
      unsigned gpu_id,
      std::vector<std::shared_ptr<SmartInferRequest>> &local_request_queue);
  void gpu_local_batch_schedule(
      unsigned gpu_id,
      std::vector<std::shared_ptr<SmartInferRequest>> &local_request_queue);
  void do_schedule();
  void decide_replication();

  void init_estimates(unsigned model_id, BatchedModelState &state) {
    execution_profiler[model_id] = SmartExecutionProfiler();
    weights_profiler[model_id] = SmartWeightsProfiler();

    execution_profiler[model_id].set_batch_sizes(state.supported_batch_sizes);
    execution_profiler[model_id].set_estimates(state.exec_duration);

    unsigned window_size = 100;
    execution_profiler[model_id].set_window_size(window_size);
    weights_profiler[model_id].estimate = state.weights_transfer_duration;
  }

  void set_estimates(unsigned model_id, unsigned batch_size,
                     uint64_t exec_latency, unsigned freq) {
    mtx_profiler.lock();
    execution_profiler[model_id].insert(batch_size, exec_latency, freq);
    mtx_profiler.unlock();
  }

  uint64_t get_latency_estimate(unsigned model_id, unsigned batch_size) {
    mtx_profiler.lock();
    uint64_t wcet_estimate =
        execution_profiler[model_id].get_latency_estimate(batch_size);
    mtx_profiler.unlock();
    return wcet_estimate;
  }

  uint64_t get_weights_load_estimate(unsigned model_id) {
    return weights_profiler[model_id].get_estimate();
  }

  // runs after the very first initialization, we init the system state and
  // model stats using the passed info
  void start(std::vector<network::controller::WorkerConnection *> workers,
             ClockworkState &state) {
    std::string action_telemetry_file = "./controller_action_log.csv";
    logger = ControllerActionTelemetry::log_and_summarize(action_telemetry_file,
                                                          1000000000UL);

    // initializing
    this->workers = workers;
    action_id_seed = 10000;
    global_request_id = 1000;
    slo = 100000000ULL;  // 100ms

    // -- gpu -> worker_idx, gpu_idx
    DEBUG_PRINT("Initializing GPU instances");
    unsigned gpu_id = 0;
    for (auto &worker : state.workers) {
      for (auto &gpu : worker.gpus) {
        gpu_map[gpu_id] = std::make_shared<SmartGPU>(
            SmartGPU(gpu_id, worker.id, gpu.id, gpu.weights_cache_total_pages));
        gpu_list.push_back(gpu_map[gpu_id]);

        std::cout << "gpu " << gpu_id << " : w " << worker.id << " g " << gpu.id
                  << "\n";
        gpu_id++;
      }
    }

    // -- parsing model stats
    for (auto &worker : state.workers) {
      for (auto &model : worker.models) {
        unsigned model_id = model.first;
        auto state = model.second;
        models[model_id] =
            std::make_shared<SmartModel>(SmartModel(model_id, state));
        init_estimates(model_id, state);
      }
      break;  // assuming all the models are pre-loaded on all the workers
    }
    scheduler_thread = std::thread(&SmartScheduler::do_schedule, this);
  }

  void submit_actions(
      unsigned worker_id,
      std::vector<std::shared_ptr<workerapi::Action>> &actions) {
    DEBUG_PRINT("Submit action list ");
    workers[worker_id]->sendActions(actions);
  }

  // just adds the request to the global request queue and saves the callback.
  // the scheduling happens at do_schedule()
  void clientInfer(
      clientapi::InferenceRequest &request,
      std::function<void(clientapi::InferenceResponse &)> callback) {
    // temp early locking
    mtx_request_queue.lock();
    // std::cout << "[INFER] received infer request " << global_request_id <<  "
    // to m " << request.model_id << "\n";
    uint64_t request_id = ++global_request_id;
    uint64_t arrived = util::now();
    uint64_t deadline = arrived + slo;

    // add the request to the request queue
    // mtx_request_queue.lock();
    request_queue.push_back(
        std::make_shared<SmartInferRequest>(SmartInferRequest(
            request_id, arrived, deadline,
            std::make_shared<clientapi::InferenceRequest>(request))));

    // save the callback for later
    mtx_inference_callbacks.lock();
    inference_callbacks[request_id] = callback;
    mtx_inference_callbacks.unlock();

    mtx_request_queue.unlock();
  }

  void add_request_to_gpu_request_queue(
      unsigned gpu_id, std::shared_ptr<SmartInferRequest> request) {
    if (gpu_request_queue.find(gpu_id) == gpu_request_queue.end()) {
      gpu_request_queue[gpu_id] =
          std::vector<std::shared_ptr<SmartInferRequest>>();
    }
    gpu_request_queue[gpu_id].push_back(request);
  }

  static bool finish_time_compare(std::shared_ptr<SmartInferRequest> lhs,
                                  std::shared_ptr<SmartInferRequest> rhs) {
    return lhs->finish_time > rhs->finish_time;
  }

  // result from worker --> just run the action's callback
  void resultFromWorker(std::shared_ptr<workerapi::Result> result) {
    //  DEBUG_PRINT("Received result: " + result->str());
    mtx_action_callbacks.lock();
    if (action_callbacks.find(result->id) == action_callbacks.end()) {
      CHECK(false) << " couldn't find the callback for action " << result->id
                   << std::endl;
    }
    auto it = action_callbacks.find(result->id);
    std::function<void(std::shared_ptr<workerapi::Result>)> callback =
        it->second;
    action_callbacks.erase(it);
    mtx_action_callbacks.unlock();
    callback(result);
  }

  void set_telemetry_infer(unsigned worker_id,
                           std::shared_ptr<workerapi::Infer> &action);
  void set_telemetry_evict_weights(
      unsigned worker_id, std::shared_ptr<workerapi::EvictWeights> &action);
  void set_telemetry_load_weights(
      unsigned worker_id, std::shared_ptr<workerapi::LoadWeights> &action);

  void set_telemetry_infer_result(
      std::shared_ptr<workerapi::InferResult> &result);
  void set_telemetry_evict_weights_result(
      std::shared_ptr<workerapi::EvictWeightsResult> &result);
  void set_telemetry_load_weights_result(
      std::shared_ptr<workerapi::LoadWeightsResult> &result);
  void set_telemetry_error_result(
      std::shared_ptr<workerapi::ErrorResult> &result);
};

// ----- GPU BATCH SCHEDULER ----------

void SmartScheduler::gpu_local_batch_schedule(
    unsigned gpu_id,
    std::vector<std::shared_ptr<SmartInferRequest>> &local_request_queue) {}

// ----- GPU NO_BATCH SCHEDULER ----------

void SmartScheduler::gpu_local_schedule(
    unsigned gpu_id,
    std::vector<std::shared_ptr<SmartInferRequest>> &local_request_queue) {
  std::vector<uint64_t> drop_list;
  std::vector<std::shared_ptr<SmartInferRequest>> stash;
  std::set<uint64_t> conflicting_requests;

  gpu_map[gpu_id]->gpu_idle_at = std::max<uint64_t>(
      gpu_map[gpu_id]->gpu_idle_at, util::now() + NETWORK_TRANSFER_DELAY);

  // STEP: early proning --- drop the requests if their deadline is already
  // passed or we can't make it to the deadline or not enough slack to load
  // the model
  unsigned drop_count_tmp = 0;
  for (unsigned i = 0; i < local_request_queue.size(); i++) {
    unsigned model_id = local_request_queue[i]->request_ptr->model_id;
    uint64_t execution_duration = get_latency_estimate(model_id, 1);

    // we set the earliest, start and finish times while we're iterating over
    // the local_request_queue
    local_request_queue[i]->earliest = std::max<uint64_t>(
        gpu_map[gpu_id]->gpu_idle_at,
        models[model_id]->weights_available_at[gpu_id]);
    local_request_queue[i]->finish_time = local_request_queue[i]->deadline;
    local_request_queue[i]->start_time =
        local_request_queue[i]->deadline - execution_duration;

    if (local_request_queue[i]->earliest + execution_duration >
        local_request_queue[i]
            ->deadline) {  // if the infer request cannot be done by any means,
                           // even if sent to the gpu right now
      drop_list.push_back(local_request_queue[i]->id);  // add to the drop_list
      drop_count_tmp++;
      if (model_drop_count.find(model_id) ==
          model_drop_count.end()) {  // keep the drop count of each model, so we
                                     // would decide if we want to load another
                                     // instance to alieviate the load
        model_drop_count[model_id] = 1;
      } else {
        model_drop_count[model_id]++;
      }
    }
  }

  // early dropping the requests that cannot be scheduled
  for (auto &request_id : drop_list) {
    // std::cout << "dropping " << request_id << " ...\n";
    for (unsigned i = 0; i < local_request_queue.size(); i++) {
      if (local_request_queue[i]->id == request_id) {
        clientapi::InferenceResponse response;
        response.header.user_request_id =
            local_request_queue[i]->request_ptr->header.user_request_id;
        response.header.status = clockworkError;
        response.header.message = "dropped before execution";
        response.model_id = local_request_queue[i]->request_ptr->model_id;
        response.output_size = 0;
        response.output = nullptr;
        mtx_inference_callbacks.lock();
        auto callback = inference_callbacks[request_id];
        mtx_inference_callbacks.unlock();
        callback(response);
        inference_callbacks.erase(inference_callbacks.find(request_id));

        local_request_queue.erase(
            local_request_queue.begin() +
            i);  // remove the request from the request queue

        break;
      }
    }
  }

  //   std::cout << "before initial placement ... \n";

  unsigned stash_size_prev = stash.size();
  unsigned stash_size = stash.size();
  do {
    // STEP: initial placement ---
    sort(local_request_queue.begin(), local_request_queue.end(),
         finish_time_compare);  // sort based on finish_time

    if (local_request_queue.size() > 1) {
      for (int i = local_request_queue.size() - 1; i > 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
          //   std::cout << "i: " << i << " j: " << j << "\n";
          if (i != j && !(local_request_queue[j]->finish_time <=
                          local_request_queue[i]->start_time)) {
            conflicting_requests.insert(local_request_queue[i]->id);
            conflicting_requests.insert(local_request_queue[j]->id);
          }
        }
      }
    }

    // STEP: take out the conflicting requests
    for (auto request_id : conflicting_requests) {
      for (unsigned i = 0;; i++) {
        if (i >= local_request_queue.size()) {
          break;
        }
        if (local_request_queue[i]->id == request_id) {
          stash.push_back(local_request_queue[i]);
          local_request_queue.erase(local_request_queue.begin() + i);
          break;
        }
      }
    }

    // STEP: resolve round

    std::vector<uint64_t> to_remove_from_stash;
    unsigned stash_size = stash.size();

    for (unsigned i = 0; i < stash.size(); i++) {
      // STEP: placing at the tail
      if (local_request_queue.size() == 0) {
        local_request_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i]->id);
        continue;
      } else if ((local_request_queue.size() > 0) &&
                 (stash[i]->earliest >=
                  local_request_queue[local_request_queue.size() - 1]
                      ->finish_time) &&
                 (stash[i]->deadline >=
                  stash[i]->start_time +
                      get_latency_estimate(stash[i]->request_ptr->model_id,
                                           1))) {  // start time after
        // finish? or local_queue
        // empty? OK, put it at the tail
        // add to the tail
        // mark to remove from the stash
        stash[i]->start_time =
            local_request_queue[local_request_queue.size() - 1]->finish_time;
        stash[i]->finish_time =
            stash[i]->start_time +
            get_latency_estimate(stash[i]->request_ptr->model_id, 1);

        local_request_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i]->id);
        sort(local_request_queue.begin(), local_request_queue.end(),
             finish_time_compare);  // sort based on finish_time
        continue;
      }

      // STEP: Place in a hole
      bool placed_in_a_hole = false;
      for (int j = local_request_queue.size() - 1; j > 0; j--) {
        if (local_request_queue[j - 1]->finish_time +
                    get_latency_estimate(stash[i]->request_ptr->model_id, 1) <=
                stash[i]->deadline &&
            (stash[i]->earliest <=
             local_request_queue[j]->start_time -
                 get_latency_estimate(stash[i]->request_ptr->model_id, 1)) &&
            (local_request_queue[j]->start_time -
                 local_request_queue[j - 1]->finish_time >=
             get_latency_estimate(stash[i]->request_ptr->model_id,
                                  1))) {  // if the request fits
                                          // between two
                                          // scheduled requests
          stash[i]->finish_time = local_request_queue[j]->start_time;
          stash[i]->start_time =
              local_request_queue[j]->start_time -
              get_latency_estimate(stash[i]->request_ptr->model_id, 1);
          local_request_queue.push_back(stash[i]);
          to_remove_from_stash.push_back(stash[i]->id);
          sort(local_request_queue.begin(), local_request_queue.end(),
               finish_time_compare);
          placed_in_a_hole = true;
          break;
        }
      }
      // STEP: Place at the head
      if (!placed_in_a_hole &&
          local_request_queue[0]->start_time >=
              stash[i]->earliest +
                  get_latency_estimate(stash[i]->request_ptr->model_id, 1)) {
        stash[i]->finish_time = local_request_queue[0]->start_time;
        stash[i]->start_time =
            local_request_queue[0]->start_time -
            get_latency_estimate(stash[i]->request_ptr->model_id, 1);
        local_request_queue.push_back(stash[i]);
        to_remove_from_stash.push_back(stash[i]->id);
        sort(local_request_queue.begin(), local_request_queue.end(),
             finish_time_compare);
        continue;
      }
    }

    // STEP: remove scheduled requests from the stash
    for (auto request_id : to_remove_from_stash) {
      for (int i = 0;; i++) {
        if (i >= stash.size()) {
          break;
        }
        if (stash[i]->id == request_id) {
          stash.erase(stash.begin() + i);
        }
      }
    }

    // STEP: compressing the schedule
    if (local_request_queue.size() > 0) {
      local_request_queue[0]->start_time = std::max<uint64_t>(
          local_request_queue[0]->earliest,
          gpu_map[gpu_id]->gpu_idle_at);  // shift the first element to the
                                        // earliest time possible
      local_request_queue[0]->finish_time =
          local_request_queue[0]->start_time +
          get_latency_estimate(local_request_queue[0]->request_ptr->model_id,
                               1);
      for (unsigned i = 1; i < local_request_queue.size(); i++) {
        local_request_queue[i]->start_time =
            std::max<uint64_t>(local_request_queue[i]->earliest,
                               local_request_queue[i - 1]->finish_time);
        local_request_queue[i]->finish_time =
            local_request_queue[i]->start_time +
            get_latency_estimate(local_request_queue[i]->request_ptr->model_id,
                                 1);
      }
    }

    stash_size = stash.size();
  } while (stash_size_prev != stash_size);

  // STEP append all the remaining stashed to the drop_list
  // drop all stashed which can't be started in the current epoch
  for (auto request_item : stash) {
    if (request_item->deadline <=
        gpu_map[gpu_id]->gpu_idle_at + SCHEDULING_EPOCH) {
      clientapi::InferenceResponse response;
      response.header.user_request_id =
          request_item->request_ptr->header.user_request_id;
      response.header.status = clockworkError;
      response.header.message = "dropped before execution";
      response.output_size = 0;
      response.output = nullptr;

      mtx_inference_callbacks.lock();
      auto callback = inference_callbacks[request_item->id];

      inference_callbacks.erase(inference_callbacks.find(request_item->id));
      mtx_inference_callbacks.unlock();
      callback(response);
      // add to drop_list
      drop_list.push_back(request_item->id);
      // remove the callback
      break;
    } else {
      unsigned model_id = request_item->request_ptr->model_id;
      if (model_drop_count.find(model_id) == model_drop_count.end()) {
        model_drop_count[model_id] = 1;
      } else {
        model_drop_count[model_id]++;
      }
    }
  }

  std::vector<std::shared_ptr<workerapi::Action>> actions;

  // STEP: create infer actions
  unsigned index;
  for (index = 0; index < local_request_queue.size(); index++) {
    if (local_request_queue[index]->start_time >
        gpu_map[gpu_id]->gpu_idle_at + SCHEDULE_AHEAD) {
      break;
    }
    auto infer = std::make_shared<workerapi::Infer>();
    uint64_t request_id = local_request_queue[index]->id;
    unsigned user_request_id =
        local_request_queue[index]->request_ptr->header.user_request_id;
    drop_list.push_back(request_id);
    infer->id = ++action_id_seed;
    infer->model_id = local_request_queue[index]->request_ptr->model_id;
    infer->gpu_id = gpu_map[gpu_id]->gpu_index;
    infer->batch_size = 1;
    infer->earliest = local_request_queue[index]->start_time;
    infer->latest = local_request_queue[index]->start_time + INFER_SLACK;

    // update the gpu timings
    gpu_map[gpu_id]->gpu_idle_at = local_request_queue[index]->finish_time;

    // update LRU model on the GPU
    unsigned model_id = infer->model_id;
    gpu_map[gpu_id]->update_lru(infer->model_id);

    auto infer_action_complete =
        [this, request_id, model_id,
         user_request_id](std::shared_ptr<workerapi::Result> result) {
          if (auto infer_result =
                  std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
            set_telemetry_infer_result(infer_result);

            set_estimates(model_id, 1, infer_result->exec.duration,
                          infer_result->gpu_clock);

            mtx_inference_callbacks.lock();
            auto callback = inference_callbacks[request_id];
            inference_callbacks.erase(inference_callbacks.find(request_id));
            mtx_inference_callbacks.unlock();

            clientapi::InferenceResponse response;
            response.header.user_request_id = user_request_id;
            response.header.status = clockworkSuccess;
            response.header.message = "";
            response.output_size = infer_result->output_size;
            response.output = infer_result->output;
            response.model_id = model_id;
            response.batch_size = 1;
            callback(response);

          } else {
            std::string error_message = "Internal Controller Error";

            if (auto error =
                    std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
              error_message = error->message;
              set_telemetry_error_result(error);
            }

            mtx_inference_callbacks.lock();
            auto callback = inference_callbacks[request_id];
            inference_callbacks.erase(inference_callbacks.find(request_id));
            mtx_inference_callbacks.unlock();

            clientapi::InferenceResponse response;
            response.header.user_request_id = user_request_id;
            response.header.status = clockworkError;
            response.header.message = error_message;
            response.output_size = 0;
            response.output = nullptr;
            response.model_id = model_id;
            callback(response);
          }
        };
    mtx_action_callbacks.lock();
    action_callbacks[infer->id] = infer_action_complete;
    mtx_action_callbacks.unlock();

    set_telemetry_infer(gpu_map[gpu_id]->worker_id, infer);

    actions.push_back(infer);
  }

  // STEP: send actions to the worker

  if (actions.size() > 0) {
    workers[gpu_map[gpu_id]->worker_id]->sendActions(actions);
    local_request_queue.erase(local_request_queue.begin(),
                              local_request_queue.begin() + index);
  }

  // STEP: remove the scheduled requests from the request queue
  for (auto req_id : drop_list) {
    for (unsigned idx = 0; idx < request_queue.size(); idx++) {
      if (request_queue[idx]->id == req_id) {
        request_queue.erase(request_queue.begin() + idx);
      }
    }
  }
}

// utility comparator function to pass to the sort() module
bool sortByVal(const std::pair<unsigned, uint64_t> &a,
               const std::pair<unsigned, uint64_t> &b) {
  return (a.second < b.second);
}

void SmartScheduler::decide_replication() {
  std::set<unsigned> queued_models;
  std::map<unsigned, unsigned> model_queued_load;  // model_id -> incoming_load

  // initial incoming load check
  for (auto &request_entry : request_queue) {
    unsigned model_id = request_entry->request_ptr->model_id;
    queued_models.insert(model_id);
    if (model_queued_load.find(model_id) == model_queued_load.end()) {
      model_queued_load[model_id] = 1;
    } else {
      model_queued_load[model_id]++;
    }
  }

  std::map<unsigned, unsigned>
      models_to_replicate;  // model_id -> num_new_replicas

  // TODO: this metric of drop counts per scheduling attempt doesn't seem to
  // be as good as expected, because there isn't enough dropped request per
  // model per each schedule to trigger the replication another idea: based on
  // the sum of drop_counts within the last N schedules (= like 10, 10 * 3ms =
  // 30ms, it would take us 30ms to detect a high load on a model?)
  for (auto &model_drop_count_item : model_drop_count) {
    unsigned model_id = model_drop_count_item.first;
    unsigned drop_count = model_drop_count_item.second;
    unsigned total_gpu_count = gpu_map.size();
    unsigned current_assigned_gpus_count = model_gpu_map[model_id].size();
    unsigned replication_sensitivity =
        REPLICATION_SENSITIVITY;  // it means add one replica per
                                  // [replication_param] dropped requests, the
                                  // smaller this parameter is, the more
                                  // aggressive. over sensitive replication
                                  // could hurt the overal performance it gets
                                  // at replication
    unsigned new_replicas_to_load =
        std::min<unsigned>(total_gpu_count - current_assigned_gpus_count,
                           drop_count / replication_sensitivity);  //
    if (new_replicas_to_load > 0) {
      // deal breaker: if the model is already being loaded on any gpu, don't
      // plan any further replication for it
      bool currently_loading = false;
      for (auto &weights_available_at :
           models[model_id]->weights_available_at) {
        if (weights_available_at.second + 10000000 >
            util::now()) {  // + 10ms to wait before initiating any new
                            // replication
          currently_loading = true;
          break;
        }
      }
      if (currently_loading) {
        continue;
      }

      models_to_replicate[model_id] = new_replicas_to_load;
    }
  }

  std::map<unsigned, uint64_t>
      estimated_gpu_load;  // gpu_id -> estimated_idle_at

  // init the gpu load estimates
  for (auto &gpu_load_item : gpu_map) {
    unsigned gpu_id = gpu_load_item.first;
    estimated_gpu_load[gpu_id] = std::max<uint64_t>(
        gpu_map[gpu_id]->gpu_idle_at, util::now() + NETWORK_TRANSFER_DELAY);
  }

  std::map<unsigned, std::map<unsigned, std::vector<unsigned>>>
      replication_plan;  // gpu_id -> [ model_id, [victims] ]

  for (auto &model_to_replicate :
       models_to_replicate) {  // outer loop, iterating over models_to_load to
                               // find a gpu for each

    unsigned model_to_replicate_id = model_to_replicate.first;

    // calculate the estimated gpu loads
    for (auto &model_load_element :
         model_queued_load) {  // iterate all the models

      unsigned model_id = model_load_element.first;
      unsigned queued_load = model_load_element.second;

      for (auto &model_gpu_element : model_gpu_map[model_id]) {
        unsigned gpu_id = model_gpu_element;
        estimated_gpu_load[gpu_id] +=
            (get_latency_estimate(model_id, 1) * queued_load /
             (model_gpu_map[model_id].size() +
              1));  // distribute the load on all the gpus serving
                    // the target model
      }
    }

    std::vector<std::pair<unsigned, uint64_t>> sorted_gpu_loads;
    for (auto &gpu_load_estimate_item : estimated_gpu_load) {
      sorted_gpu_loads.push_back(std::make_pair(gpu_load_estimate_item.first,
                                                gpu_load_estimate_item.second));
    }
    sort(sorted_gpu_loads.begin(), sorted_gpu_loads.end(), sortByVal);

    for (auto &sorted_gpu_load_item : sorted_gpu_loads) {
      unsigned gpu_id = sorted_gpu_load_item.first;
      if (std::find(model_gpu_map[model_to_replicate_id].begin(),
                    model_gpu_map[model_to_replicate_id].end(),
                    gpu_id) == model_gpu_map[model_to_replicate_id].end()) {
        // if the model is not loaded on this gpu, add it to the replication
        // plan
        if (replication_plan.find(gpu_id) == replication_plan.end()) {
          replication_plan[gpu_id] =
              std::map<unsigned, std::vector<unsigned>>();
        }
        replication_plan[gpu_id][model_to_replicate_id] =
            std::vector<unsigned>();

        // find the victims

        while (gpu_map[gpu_id]->available_pages <=
               models[model_to_replicate_id]
                   ->state.num_weights_pages) {  // evict until there's
                                                 // enough space
          // evict
          unsigned victim_model = gpu_map[gpu_id]->lru_loaded_models.back();
          while (queued_models.find(victim_model) !=
                 queued_models.end()) {  // if the model is being requested put
                                         // it back to the lru list at the head
            gpu_map[gpu_id]->lru_loaded_models.insert(
                gpu_map[gpu_id]->lru_loaded_models.begin(), victim_model);
            victim_model = gpu_map[gpu_id]->lru_loaded_models.back();
          }

          replication_plan[gpu_id][model_to_replicate_id].push_back(
              victim_model);
          gpu_map[gpu_id]->available_pages +=
              models[victim_model]->state.num_weights_pages;

          auto pos = std::find(model_gpu_map[victim_model].begin(),
                               model_gpu_map[victim_model].end(), gpu_id);
          model_gpu_map[victim_model].erase(pos);
          gpu_map[gpu_id]->evict_model(victim_model);

          if (model_gpu_map[victim_model].size() ==
              0) {  // if there's no loaded instance in the entire system
            model_gpu_map.erase(model_gpu_map.find(
                victim_model));  // delete the whole entry of model_gpu_map
            global_model_cache_stat.erase(global_model_cache_stat.find(
                victim_model));  // delete the global_model_cache entry
          }
        }
      }
    }
  }

  // -- by now we have the model gpu id -> model evict lists
  // STEP: make evict / load actions

  auto on_complete = [this](std::shared_ptr<workerapi::Result> result) {
    if (auto evict_weights_result =
            std::dynamic_pointer_cast<workerapi::EvictWeightsResult>(result)) {
      set_telemetry_evict_weights_result(evict_weights_result);
    } else if (auto load_weights_result =
                   std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(
                       result)) {
      set_telemetry_load_weights_result(load_weights_result);
    } else if (auto error =
                   std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
      set_telemetry_error_result(error);
      CHECK(false) << "Load/Evict weights failed: " << error->message;
    } else {
      CHECK(false) << "Load/Evict weights failed: Internal Controller Error";
    }
  };

  for (auto &gpu_pci_ops_item : replication_plan) {
    unsigned gpu_id = gpu_pci_ops_item.first;
    gpu_map[gpu_id]->pci_idle_at = std::max<uint64_t>(
        gpu_map[gpu_id]->pci_idle_at, util::now() + NETWORK_TRANSFER_DELAY);

    std::vector<std::shared_ptr<workerapi::Action>> evict_load_actions;

    for (auto &model_pci_ops_item : gpu_pci_ops_item.second) {
      unsigned model_id = model_pci_ops_item.first;

      for (auto &victim_model_id : model_pci_ops_item.second) {
        auto evict_weights_action = std::make_shared<workerapi::EvictWeights>();
        evict_weights_action->id = ++action_id_seed;
        evict_weights_action->model_id = victim_model_id;
        evict_weights_action->gpu_id = gpu_map[gpu_id]->gpu_index;
        evict_weights_action->earliest = gpu_map[gpu_id]->pci_idle_at;
        evict_weights_action->latest = gpu_map[gpu_id]->pci_idle_at +
                                       PCI_AVERAGE_SLACK;  // latest to start
        evict_load_actions.push_back(evict_weights_action);
        gpu_map[gpu_id]->pci_idle_at +=
            WEIGHTS_EVICT_LATENCY + PCI_AVERAGE_SLACK;
        models[victim_model_id]->weights_available_at.erase(
            models[victim_model_id]->weights_available_at.find(gpu_id));

        mtx_action_callbacks.lock();
        action_callbacks[evict_weights_action->id] = on_complete;
        mtx_action_callbacks.unlock();

        set_telemetry_evict_weights(gpu_map[gpu_id]->worker_id,
                                    evict_weights_action);
      }

      auto load_weights_action = std::make_shared<workerapi::LoadWeights>();
      load_weights_action->id = ++action_id_seed;
      load_weights_action->model_id = model_id;
      load_weights_action->gpu_id = gpu_map[gpu_id]->gpu_index;
      load_weights_action->earliest = gpu_map[gpu_id]->pci_idle_at;
      load_weights_action->latest =
          gpu_map[gpu_id]->pci_idle_at + PCI_AVERAGE_SLACK;  // 1ms slack
      evict_load_actions.push_back(load_weights_action);
      gpu_map[gpu_id]->pci_idle_at +=
          get_weights_load_estimate(model_id) + PCI_AVERAGE_SLACK;
      models[model_id]->weights_available_at[gpu_id] =
          gpu_map[gpu_id]->pci_idle_at;
      model_gpu_map[model_id].push_back(gpu_id);
      gpu_map[gpu_id]->loaded_models.insert(model_id);

      mtx_action_callbacks.lock();
      action_callbacks[load_weights_action->id] = on_complete;
      mtx_action_callbacks.unlock();

      set_telemetry_load_weights(gpu_map[gpu_id]->worker_id,
                                 load_weights_action);
    }

    workers[gpu_map[gpu_id]->worker_id]->sendActions(evict_load_actions);
  }
}

// the scheduler thread runs this function every EPOCH
void SmartScheduler::do_schedule() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(SCHEDULING_EPOCH));
    if (request_queue.size() == 0) {
      continue;
    }
    // conservative early locking of the global request queue
    mtx_request_queue.lock();

    // STEP:  do a quick pass on the request queue to see if we need to load
    // any models

    std::set<unsigned> queued_models;
    std::map<unsigned, unsigned> models_to_load;  // model_id -> gpu_id
    std::map<unsigned, unsigned>
        model_queued_load;  // model_id -> incoming_load

    // std::cout << "Reading the request queue ...\n";
    for (auto &request_entry : request_queue) {
      unsigned model_id = request_entry->request_ptr->model_id;
      queued_models.insert(model_id);
      if (global_model_cache_stat.find(model_id) ==
          global_model_cache_stat.end()) {  // if the model is hot on any gpu
                                            // nor scheduled to be loaded
        models_to_load[model_id] = UINT_MAX;
      }
      // we'd estimate the incoming load in the request queue, no matter the
      // model is already loaded or not
      if (model_queued_load.find(model_id) == model_queued_load.end()) {
        model_queued_load[model_id] = 1;
      } else {
        model_queued_load[model_id]++;
      }
    }

    // STEP: which gpu should we load the new model on? pick the least loaded
    // gpu
    // calculating estimated load

    std::map<unsigned, uint64_t>
        estimated_gpu_load;  // gpu_id -> estimated_idle_at

    // estimated gpu load: the gpu load is estimated by the
    // time it's envisioned to finish all the current and incoming tasks. the
    // smaller this number is, the less loaded is the gpu

    // init the gpu load estimates
    for (auto &gpu_load_item : gpu_map) {
      unsigned gpu_id = gpu_load_item.first;
      estimated_gpu_load[gpu_id] = std::max<uint64_t>(
          gpu_map[gpu_id]->gpu_idle_at, util::now() + NETWORK_TRANSFER_DELAY);
    }

    //
    for (auto &model_to_load :
         models_to_load) {  // outer loop, iterating over models_to_load to
                            // find a gpu for each

      for (auto &model_load_element :
           model_queued_load) {  // iterate all the models

        unsigned model_id = model_load_element.first;
        unsigned queued_load = model_load_element.second;

        if (global_model_cache_stat.find(model_id) !=
            global_model_cache_stat.end()) {  // if the model is already loaded
          for (auto &model_gpu_element :
               model_gpu_map[model_id]) {  // iterate over
            unsigned gpu_id = model_gpu_element;

            estimated_gpu_load[gpu_id] +=
                (get_latency_estimate(model_id, 1) * queued_load /
                 (model_gpu_map[model_id].size() +
                  1));  // distribute the load on all the gpus serving
                        // the target model
          }
        } else {  // if the model is a not loaded nor scheduled to be loaded
                  // yet. In other words: adding the estimated future load for
                  // the models which are not already loaded
          for (auto &to_load_model : models_to_load) {
            unsigned future_gpu_id = to_load_model.second;
            unsigned to_load_model_id = to_load_model.first;

            if (to_load_model_id == model_id && future_gpu_id != UINT_MAX) {
              estimated_gpu_load[future_gpu_id] +=
                  (get_weights_load_estimate(model_id) +
                   get_latency_estimate(model_id, 1) * queued_load /
                       (model_gpu_map[model_id].size() + 1));
            }
          }
        }
      }

      // STEP: finding the least loaded gpu
      unsigned target_gpu_id = 0;
      uint64_t gpu_min_load = ULLONG_MAX;

      for (auto &gpu_load : estimated_gpu_load) {
        unsigned gpu_id = gpu_load.first;
        uint64_t estimated_load = gpu_load.second;
        if (estimated_load < gpu_min_load && gpu_id != UINT_MAX) {
          gpu_min_load = estimated_load;
          target_gpu_id = gpu_id;
        }
      }
      model_to_load.second = target_gpu_id;
      estimated_gpu_load[target_gpu_id] +=
          get_weights_load_estimate(model_to_load.first) +
          model_queued_load[model_to_load.first] *
              get_latency_estimate(model_to_load.first, 1);
    }

    // STEP: check if there is enough space on the target gpus, otherwise
    // evict
    std::map<unsigned, std::map<unsigned, std::vector<unsigned>>>
        gpu_model_load_evict;  // gpu_id -> to_load_model, [to_evict_models]

    for (auto &model_to_load : models_to_load) {
      unsigned model_id = model_to_load.first;
      unsigned gpu_id = model_to_load.second;

      if (gpu_model_load_evict.find(gpu_id) == gpu_model_load_evict.end()) {
        gpu_model_load_evict[gpu_id] =
            std::map<unsigned, std::vector<unsigned>>();
      }
      gpu_model_load_evict[gpu_id][model_id] = std::vector<unsigned>();

      while (
          gpu_map[gpu_id]->available_pages <=
          models[model_id]->state.num_weights_pages) {  // evict until there's
                                                        // enough space
        // evict
        unsigned victim_model = gpu_map[gpu_id]->lru_loaded_models.back();
        while (queued_models.find(victim_model) !=
               queued_models.end()) {  // if the model is being requested put
                                       // it back to the lru list at the head
          gpu_map[gpu_id]->lru_loaded_models.insert(
              gpu_map[gpu_id]->lru_loaded_models.begin(), victim_model);
          victim_model = gpu_map[gpu_id]->lru_loaded_models.back();
        }

        gpu_model_load_evict[gpu_id][model_id].push_back(victim_model);
        gpu_map[gpu_id]->available_pages +=
            models[victim_model]->state.num_weights_pages;

        auto pos = std::find(model_gpu_map[victim_model].begin(),
                             model_gpu_map[victim_model].end(), gpu_id);
        model_gpu_map[victim_model].erase(pos);
        gpu_map[gpu_id]->evict_model(victim_model);

        if (model_gpu_map[victim_model].size() ==
            0) {  // if there's no loaded instance in the entire system
          model_gpu_map.erase(model_gpu_map.find(
              victim_model));  // delete the whole entry of model_gpu_map
          global_model_cache_stat.erase(global_model_cache_stat.find(
              victim_model));  // delete the global_model_cache entry
        }
      }
      if (model_gpu_map.find(model_id) == model_gpu_map.end()) {
        model_gpu_map[model_id] = std::vector<unsigned>();
      }
      global_model_cache_stat.insert(model_id);
      gpu_map[gpu_id]->loaded_models.insert(model_id);
      model_gpu_map[model_id].push_back(gpu_id);
    }

    // -- by now we have the model gpu id -> model evict lists
    // STEP: make evict / load actions

    auto on_complete = [this](std::shared_ptr<workerapi::Result> result) {
      if (auto evict_weights_result =
              std::dynamic_pointer_cast<workerapi::EvictWeightsResult>(
                  result)) {
        set_telemetry_evict_weights_result(evict_weights_result);
      } else if (auto load_weights_result =
                     std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(
                         result)) {
        set_telemetry_load_weights_result(load_weights_result);
      } else if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(
                     result)) {
        set_telemetry_error_result(error);
        // TODO: mark the model as unavailable if loading failed
        // TODO: mark the model as available if eviction failed
        CHECK(false) << "Load/Evict weights failed: " << error->message;
      } else {
        CHECK(false) << "Load/Evict weights failed: Internal Controller Error";
      }
    };

    for (auto &gpu_pci_ops_item : gpu_model_load_evict) {
      unsigned gpu_id = gpu_pci_ops_item.first;
      gpu_map[gpu_id]->pci_idle_at = std::max<uint64_t>(
          gpu_map[gpu_id]->pci_idle_at, util::now() + NETWORK_TRANSFER_DELAY);

      std::vector<std::shared_ptr<workerapi::Action>> evict_load_actions;

      for (auto &model_pci_ops_item : gpu_pci_ops_item.second) {
        unsigned model_id = model_pci_ops_item.first;

        for (auto &victim_model_id : model_pci_ops_item.second) {
          auto evict_weights_action =
              std::make_shared<workerapi::EvictWeights>();
          evict_weights_action->id = ++action_id_seed;
          evict_weights_action->model_id = victim_model_id;
          evict_weights_action->gpu_id = gpu_map[gpu_id]->gpu_index;
          evict_weights_action->earliest = gpu_map[gpu_id]->pci_idle_at;
          evict_weights_action->latest = gpu_map[gpu_id]->pci_idle_at +
                                         PCI_AVERAGE_SLACK;  // latest to start
          evict_load_actions.push_back(evict_weights_action);
          gpu_map[gpu_id]->pci_idle_at +=
              WEIGHTS_EVICT_LATENCY + PCI_AVERAGE_SLACK;
          models[victim_model_id]->weights_available_at.erase(
              models[victim_model_id]->weights_available_at.find(gpu_id));

          set_telemetry_evict_weights(gpu_map[gpu_id]->worker_id,
                                      evict_weights_action);
          mtx_action_callbacks.lock();
          action_callbacks[evict_weights_action->id] = on_complete;
          mtx_action_callbacks.unlock();
        }

        auto load_weights_action = std::make_shared<workerapi::LoadWeights>();
        load_weights_action->id = ++action_id_seed;
        load_weights_action->model_id = model_id;
        load_weights_action->gpu_id = gpu_map[gpu_id]->gpu_index;
        load_weights_action->earliest = gpu_map[gpu_id]->pci_idle_at;
        load_weights_action->latest =
            gpu_map[gpu_id]->pci_idle_at + PCI_AVERAGE_SLACK;  // 1ms slack
        evict_load_actions.push_back(load_weights_action);
        gpu_map[gpu_id]->pci_idle_at +=
            get_weights_load_estimate(model_id) + PCI_AVERAGE_SLACK;
        models[model_id]->weights_available_at[gpu_id] =
            gpu_map[gpu_id]->pci_idle_at;

        set_telemetry_load_weights(gpu_map[gpu_id]->worker_id,
                                   load_weights_action);

        mtx_action_callbacks.lock();
        action_callbacks[load_weights_action->id] = on_complete;
        mtx_action_callbacks.unlock();
      }
      workers[gpu_map[gpu_id]->worker_id]->sendActions(evict_load_actions);
    }

    // STEP: assigning requests to the gpus

    std::map<unsigned, unsigned>
        model_multi_gpu_placement_idx;  // model_id ->
                                        // current_assigned_gpu_index
    for (auto &request_entry : request_queue) {
      auto request = request_entry->request_ptr;
      unsigned model_id = request->model_id;
      unsigned gpu_id;
      if (model_gpu_map.find(model_id) != model_gpu_map.end()) {
        if (model_gpu_map[model_id].size() ==
            1) {  // if only one gpu assigned to this model
          gpu_id =
              model_gpu_map[model_id]
                           [0];  // assign the request to the only gpu available
        } else {  // if more than one gpu is assigned to this model
          if (model_multi_gpu_placement_idx.find(model_id) ==
              model_multi_gpu_placement_idx.end()) {
            model_multi_gpu_placement_idx[model_id] =
                0;  // start by the first one
          } else {
            model_multi_gpu_placement_idx[model_id] =
                (model_multi_gpu_placement_idx[model_id] + 1) %
                model_gpu_map[model_id]
                    .size();  // iterate over assigned gpu idxs
          }
          gpu_id = model_multi_gpu_placement_idx[model_id];
        }
      } else {
        CHECK(false) << "there's a problem with model loading\n";
      }
      request_entry->earliest =
          models[model_id]
              ->weights_available_at[gpu_id];  // the earliest a request can be
                                               // executed is after the weights
                                               // are available
      add_request_to_gpu_request_queue(gpu_id, request_entry);
    }

    // do local scheduling on each gpu queue
    for (auto &local_request_item : gpu_request_queue) {
      if (local_request_item.second.size() ==
          0) {  // if there's no request assigned to a gpu, don't bother calling
                // gpu_local_schedule
        continue;
      }
      gpu_local_schedule(local_request_item.first, local_request_item.second);
    }

    // STEP: check if we need to replicate any model that has high load
    decide_replication();

    model_drop_count.clear();
    gpu_request_queue.clear();
    mtx_request_queue.unlock();
  }
}

// --- Telemetry related functions

void SmartScheduler::set_telemetry_infer(
    unsigned worker_id, std::shared_ptr<workerapi::Infer> &action) {
  ControllerActionTelemetry *telemetry = new ControllerActionTelemetry;
  telemetry->worker_id = worker_id;
  telemetry->set(action);
  mtx_telemetry.lock();
  action_telemetry_map.insert(std::make_pair(action->id, telemetry));
  mtx_telemetry.unlock();
}

void SmartScheduler::set_telemetry_evict_weights(
    unsigned worker_id, std::shared_ptr<workerapi::EvictWeights> &action) {
  ControllerActionTelemetry *telemetry = new ControllerActionTelemetry;
  telemetry->worker_id = worker_id;
  telemetry->set(action);
  mtx_telemetry.lock();
  action_telemetry_map.insert(std::make_pair(action->id, telemetry));
  mtx_telemetry.unlock();
}

void SmartScheduler::set_telemetry_load_weights(
    unsigned worker_id, std::shared_ptr<workerapi::LoadWeights> &action) {
  ControllerActionTelemetry *telemetry = new ControllerActionTelemetry;
  telemetry->worker_id = worker_id;
  telemetry->set(action);
  mtx_telemetry.lock();
  action_telemetry_map.insert(std::make_pair(action->id, telemetry));
  mtx_telemetry.unlock();
}

void SmartScheduler::set_telemetry_infer_result(
    std::shared_ptr<workerapi::InferResult> &result) {
  mtx_telemetry.lock();
  auto it = action_telemetry_map.find(result->id);
  ControllerActionTelemetry *telemetry = it->second;
  action_telemetry_map.erase(it);
  mtx_telemetry.unlock();
  telemetry->set(result);
  logger->log(*telemetry);
  free(telemetry);
}

void SmartScheduler::set_telemetry_load_weights_result(
    std::shared_ptr<workerapi::LoadWeightsResult> &result) {
  mtx_telemetry.lock();
  auto it = action_telemetry_map.find(result->id);
  ControllerActionTelemetry *telemetry = it->second;
  action_telemetry_map.erase(it);
  mtx_telemetry.unlock();
  telemetry->set(result);
  logger->log(*telemetry);
  free(telemetry);
}

void SmartScheduler::set_telemetry_evict_weights_result(
    std::shared_ptr<workerapi::EvictWeightsResult> &result) {
  mtx_telemetry.lock();
  auto it = action_telemetry_map.find(result->id);
  ControllerActionTelemetry *telemetry = it->second;
  action_telemetry_map.erase(it);
  mtx_telemetry.unlock();
  telemetry->set(result);
  logger->log(*telemetry);
  free(telemetry);
}

void SmartScheduler::set_telemetry_error_result(
    std::shared_ptr<workerapi::ErrorResult> &result) {
  mtx_telemetry.lock();
  auto it = action_telemetry_map.find(result->id);
  ControllerActionTelemetry *telemetry = it->second;
  action_telemetry_map.erase(it);
  mtx_telemetry.unlock();
  telemetry->set(result);
  logger->log(*telemetry);
  free(telemetry);
}

}  // namespace clockwork

#endif
