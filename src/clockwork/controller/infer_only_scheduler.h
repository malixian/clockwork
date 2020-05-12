#ifndef _CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_H_
#define _CLOCKWORK_CONTROLLER_INFER_ONLY_SCHEDULER_H_

#include "clockwork/controller/scheduler.h"
#include "clockwork/telemetry/controller_action_logger.h"
#include <atomic>


namespace clockwork {

uint64_t action_id_seed = 0;

class InferOnlyScheduler : public Scheduler {
public:
	static const uint64_t print_interval = 10000000000UL; // 10 seconds
	bool print_debug = false;
	std::mutex mutex;
	uint64_t outstanding_loads = 0;
	ControllerActionTelemetryLogger* printer;

	InferOnlyScheduler(
		std::string actions_filename = "/local/clockwork_action_log.tsv"
	) : printer(ControllerActionTelemetry::log_and_summarize(actions_filename, print_interval)) {
	}

	struct GPU;
	struct PendingInfer {
		clientapi::InferenceRequest request;
		std::function<void(clientapi::InferenceResponse&)> callback;
		GPU* assigned_gpu;
		ControllerActionTelemetry telemetry;
		uint64_t deadline;
	};
	struct GPU {
		network::controller::WorkerConnection* worker;
		unsigned gpu_id;
		unsigned worker_id;
		unsigned outstanding = 0;
		std::queue<PendingInfer*> queue;
	};

	
	unsigned max_outstanding = 4;
	std::queue<GPU*> gpus;
	std::unordered_map<uint64_t, PendingInfer*> outstanding_actions;

	// Called when model loading has completed
	virtual void start( std::vector<network::controller::WorkerConnection*> workers,
						ClockworkState &state) {
		std::lock_guard<std::mutex> lock(mutex);

		unsigned weights_cache_total_pages = state.workers[0].gpus[0].weights_cache_total_pages;
		for (auto &worker : state.workers) {
			for (auto &gpu : worker.gpus) {
				CHECK(gpu.weights_cache_total_pages == weights_cache_total_pages) << "Expect same cache size on all GPUs";
			}
		}

		for (auto &p : state.workers[0].models) {
			unsigned model_id = p.first;
			for (auto &worker : state.workers) {
				CHECK(worker.models.find(model_id) != worker.models.end()) << "Inconsistent models across workers";
			}
		}

		unsigned total_loaded = 0;
		std::vector<unsigned> models_to_load;
		for (auto &p : state.workers[0].models) {
			if (total_loaded + p.second.num_weights_pages < weights_cache_total_pages) {
				models_to_load.push_back(p.first);
				total_loaded += p.second.num_weights_pages;
			}
		}

		std::cout << "Loading " << total_loaded << " models onto GPU (" << total_loaded << " pages)" << std::endl;

		for (WorkerState &worker : state.workers) {
			std::vector<std::shared_ptr<workerapi::Action>> actions;

			for (GPUState &gpustate : worker.gpus) {
				for (unsigned &model_id : models_to_load) {
					auto load = std::make_shared<workerapi::LoadWeights>();
					load->id = action_id_seed++;
					load->gpu_id = gpustate.id;
					load->model_id = model_id;
					load->earliest = 0;
					load->latest = UINT64_MAX;
					actions.push_back(load);
				}

				GPU* gpu = new GPU();
				gpu->worker = workers[worker.id];
				gpu->worker_id = worker.id;
				gpu->gpu_id = gpustate.id;
				gpus.push(gpu);
			}

			outstanding_loads += actions.size();
			workers[worker.id]->sendActions(actions);
		}
	}

	PendingInfer* get_pending_action(std::shared_ptr<workerapi::Result> result) {
		auto it = outstanding_actions.find(result->id);
		CHECK(it != outstanding_actions.end()) 
			<< "Received result for non-existent action " << result->str();

		PendingInfer* pending = it->second;
		outstanding_actions.erase(it);
		return pending;
	}

	void sendInferSuccessToClient(
		clientapi::InferenceRequest &request, 
		std::function<void(clientapi::InferenceResponse&)> callback,
		void* output, size_t output_size)
	{
		clientapi::InferenceResponse response;
		response.header.user_request_id = request.header.user_request_id;
		response.header.status = clockworkSuccess;
		response.header.message = "";
		response.model_id = request.model_id;
		response.batch_size = request.batch_size;
		response.output_size = output_size;
		response.output = output;
		callback(response);

		if (print_debug) std::cout << "Client <--  " << response.str() << std::endl;
	}

	void sendInferErrorToClient(
		int status,
		std::string message,
		clientapi::InferenceRequest &request, 
		std::function<void(clientapi::InferenceResponse&)> callback) 
	{
		clientapi::InferenceResponse response;
		response.header.user_request_id = request.header.user_request_id;
		response.header.status = status;
		response.header.message = message;
		response.output_size = 0;
		response.output = nullptr;
		callback(response);

		if (print_debug) std::cout << "Client <--  " << response.str() << std::endl;
	}


	void sendInferActionToWorker(PendingInfer* next) {
		// Make the action
		auto infer = std::make_shared<workerapi::Infer>();
		infer->id = action_id_seed++;
		infer->model_id = next->request.model_id;
		infer->gpu_id = next->assigned_gpu->gpu_id;
		infer->batch_size = next->request.batch_size;
		infer->input = static_cast<char*>(next->request.input);
		infer->input_size = next->request.input_size;
		infer->earliest = util::now();
		infer->latest = util::now() + 10000000000UL; // 10s

		// Populate telemetry
		auto &telemetry = next->telemetry;
		telemetry.action_id = infer->id;
		telemetry.worker_id = next->assigned_gpu->worker_id;
		telemetry.gpu_id = infer->gpu_id;
		telemetry.action_type = workerapi::inferAction;
		telemetry.batch_size = infer->batch_size;
		telemetry.model_id = infer->model_id;
		telemetry.earliest = infer->earliest;
		telemetry.latest = infer->latest;
		telemetry.action_sent = util::now();

		// Save it and send
		outstanding_actions[infer->id] = next;
		next->assigned_gpu->worker->sendAction(infer);

		if (print_debug) std::cout << "Worker <--  " << infer->str() << std::endl;
	}

	void check_gpu_queue(GPU* gpu) {
		while (gpu->outstanding < max_outstanding && gpu->queue.size() > 0) {
			PendingInfer* next = gpu->queue.front();
			gpu->queue.pop();
			if (next->deadline > util::now()) {
				gpu->outstanding++;
				sendInferActionToWorker(next);
			} else {
				sendInferErrorToClient(
					clockworkTimeout, "", 
					next->request,
					next->callback
				);
				delete next;
			}
		}
	}

	void inferErrorFromWorker(std::shared_ptr<workerapi::ErrorResult> error) {
		PendingInfer* pending = get_pending_action(error);

		// Populate telemetry
		pending->telemetry.result_received = util::now();
		pending->telemetry.status = clockworkError;
		pending->telemetry.worker_duration = 0;

		pending->assigned_gpu->outstanding--;
		check_gpu_queue(pending->assigned_gpu);

		sendInferErrorToClient(error->status, error->message, 
			pending->request, pending->callback);

		printer->log(pending->telemetry);

		delete pending;
	}

	void inferSuccessFromWorker(std::shared_ptr<workerapi::InferResult> result) {
		PendingInfer* pending = get_pending_action(result);

		// Populate telemetry
		pending->telemetry.result_received = util::now();
		pending->telemetry.status = clockworkSuccess;
		pending->telemetry.worker_duration = result->exec.duration;

		pending->assigned_gpu->outstanding--;
		check_gpu_queue(pending->assigned_gpu);

		sendInferSuccessToClient(pending->request, 
			pending->callback, result->output, result->output_size);

		printer->log(pending->telemetry);

		delete pending;
	}

	// The actual controller logic once model loading has completed
	virtual void resultFromWorker(std::shared_ptr<workerapi::Result> result) {
		if (print_debug) std::cout << "Worker  --> " << result->str() << std::endl;

		std::lock_guard<std::mutex> lock(mutex);

		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			CHECK(outstanding_loads == 0) 
				<< "InferOnlyScheduler error during model load phase: " 
				<< result->str();
			inferErrorFromWorker(error);
		} else if (auto load = std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(result)) {
			CHECK(outstanding_loads > 0) 
				<< "InferOnlyScheduler received a LoadWeightsResult with no outstanding loads: " 
				<< result->str();
			outstanding_loads--;
		} else if (auto infer = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
			CHECK(outstanding_loads == 0) 
				<< "InferOnlyScheduler inconsistent state: " 
				<< result->str();
			inferSuccessFromWorker(infer);
		} else {
			CHECK(false) << "Unexpected response to LoadWeights action";
		}
	}

	virtual void clientInfer(clientapi::InferenceRequest &request, 
		std::function<void(clientapi::InferenceResponse&)> callback)
	{
		if (print_debug) std::cout << "Client  --> " << request.str() << std::endl;

		std::lock_guard<std::mutex> lock(mutex);

		if (outstanding_loads > 0) {
			sendInferErrorToClient(clockworkInitializing, 
				"InferOnlyScheduler is loading models", 
				request, 
				callback);
			return;
		}

		// Get the next GPU
		GPU* gpu = gpus.front();
		gpus.pop();
		gpus.push(gpu);

		// Add to GPU's queue
		PendingInfer* pending = new PendingInfer();
		pending->request = request;
		pending->callback = callback;
		pending->assigned_gpu = gpu;
		pending->deadline = util::now() + 100000000UL; // 100ms
		gpu->queue.push(pending);

		// Check if actions should be sent
		check_gpu_queue(gpu);
	}

};

}

#endif