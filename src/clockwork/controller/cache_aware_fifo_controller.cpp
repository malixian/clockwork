#include <atomic>
#include "clockwork/network/controller.h"
#include "clockwork/api/client_api.h"
#include "clockwork/api/worker_api.h"
#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <nvml.h>
#include <iostream>
#include "clockwork/util.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <dmlc/logging.h>
#include "cache_aware_fifo_controller.h"


using namespace clockwork;

CFifoControllerImpl::CFifoControllerImpl(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs, unsigned slo = 100) :
		Controller(client_port, worker_host_port_pairs), model_id_seed(0), action_id_seed(0), slo(slo * 1000000) { // slo conversion from millisec to nanosec
	current_worker = 0;
	for (unsigned i = 0; i < worker_host_port_pairs.size(); i++) {
		current_worker_gpu[i] = 0;
	}
}

void CFifoControllerImpl::save_callback(int id, std::function<void(std::shared_ptr<workerapi::Result>)> callback) {
	std::lock_guard<std::mutex> lock(actions_mutex);

	auto it = action_callbacks.find(id);
	CHECK(it == action_callbacks.end()) << "ID " << id << " already exists";

	action_callbacks.insert(std::make_pair(id, callback));
}

std::function<void(std::shared_ptr<workerapi::Result>)> CFifoControllerImpl::get_callback(int id) {
	std::lock_guard<std::mutex> lock(actions_mutex);

	auto it = action_callbacks.find(id);

	if (it == action_callbacks.end()) return nullptr;

	auto callback = it->second;
	action_callbacks.erase(it);
	return callback;
}

// workerapi::Controller::sendResult
void CFifoControllerImpl::sendResult(std::shared_ptr<workerapi::Result> result) {
	auto callback = get_callback(result->id);
	CHECK(callback != nullptr) << "Received result without a corresponding action: " << result->str();
	
	callback(result);
}

void CFifoControllerImpl::uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
	std::cout << "Client  -> " << request.str() << std::endl;
	clientapi::UploadModelResponse response;
	std::cout << "Client <-  " << response.str() << std::endl;
	callback(response);
}

void CFifoControllerImpl::infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
	std::cout << "Client  -> " << request.str() << std::endl;

	int user_request_id = request.header.user_request_id;
	int model_id = request.model_id;
	std::vector<std::shared_ptr<workerapi::Action>> actions;

	unsigned target_worker = current_worker;
	unsigned gpu_id = current_worker_gpu[current_worker];

	if (cached_models_stat.find(model_id) != cached_models_stat.end()) {
		target_worker = cached_models_stat[model_id].first;
		gpu_id = cached_models_stat[model_id].second;
	} else {
		current_worker_gpu[current_worker]++;
		if (current_worker_gpu[current_worker] % GPUS_PER_WORKER == 0) {
			current_worker_gpu[current_worker] = 0;
			current_worker = (current_worker + 1 ) % workers.size();
		}
		target_worker = current_worker;
		gpu_id = current_worker_gpu[current_worker];
		cached_models_stat[model_id] = std::make_pair(target_worker, gpu_id);
	}
	
	// If weights aren't loaded, send a load_weights action
	if (new_weights_available_at.find(std::make_tuple(request.model_id, target_worker, gpu_id)) == new_weights_available_at.end() ) {
		auto load_weights = std::make_shared<workerapi::LoadWeights>();
		load_weights->id = action_id_seed++;
		load_weights->model_id = request.model_id;
		load_weights->gpu_id = gpu_id;
		load_weights->earliest = util::now();
		load_weights->latest = load_weights->earliest + 100000000000UL; // 10sec slo

		auto load_weights_complete = [this, user_request_id, model_id, gpu_id] (std::shared_ptr<workerapi::Result> result) {
			std::cout << "Worker  -> " << result->str() << std::endl;

			// If the result wasn't a success, then mark the weights as unavailble again
			if (auto error_result = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
				new_weights_available_at.erase(std::make_tuple(model_id, current_worker, gpu_id));
			}
		};
		save_callback(load_weights->id, load_weights_complete);
		actions.push_back(load_weights);
		// Store when the weights will be available
		new_weights_available_at[std::make_tuple(model_id, target_worker, gpu_id)] = util::now() + 15000000UL; // Assume it'll take 7 ms to load weights
	}
	
	// Translate clientapi request into a workerapi action
	auto infer = std::make_shared<workerapi::Infer>();
	infer->id = action_id_seed++;
	infer->model_id = request.model_id;
	infer->gpu_id = gpu_id;
	infer->batch_size = request.batch_size;
	infer->input_size = request.input_size;
	infer->input = static_cast<char*>(request.input);
	

	infer->earliest = new_weights_available_at[std::make_tuple(model_id, target_worker, gpu_id)];
	infer->latest = util::now() + 100000000000UL; // 10sec slo

	// When the infer result is received, call this callback
	auto infer_complete = [this, callback, user_request_id, model_id] (std::shared_ptr<workerapi::Result> result) {
		std::cout << "Worker  -> " << result->str() << std::endl;

		// Translate workerapi result into a clientapi response
		clientapi::InferenceResponse response;
		response.header.user_request_id = user_request_id;
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			response.header.status = clockworkError;
			response.header.message = error->message;
			response.output_size = 0;
			response.output = nullptr;
		} else if (auto infer_result = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
			response.header.status = clockworkSuccess;
			response.header.message = "";
			response.model_id = model_id;
			response.batch_size = 1;
			response.output_size = infer_result->output_size;
			response.output = infer_result->output;
		} else {
			response.header.status = clockworkError;
			response.header.message = "Internal Controller Error";
			response.output_size = 0;
			response.output = nullptr;
		}
		// Send the response
		std::cout << "Client <-  " << response.str() << std::endl;
		callback(response);
	};
	save_callback(infer->id, infer_complete);
	actions.push_back(infer);

	// Send the action
	for (auto &action : actions) {
		std::cout << "Worker <-  " << action->str() << std::endl;
	}
	std::cout << "<<<<< worker: " << target_worker << "  gpu: " << gpu_id << " >>>> \n";
	workers[target_worker]->sendActions(actions);
}

/** This is a 'backdoor' API function for ease of experimentation */
void CFifoControllerImpl::evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
	std::cout << "Client  -> " << request.str() << std::endl;
	clientapi::EvictResponse response;
	std::cout << "Client <-  " << response.str() << std::endl;
	callback(response);
}

/** This is a 'backdoor' API function for ease of experimentation */
void CFifoControllerImpl::loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
	std::cout << "Client  -> " << request.str() << std::endl;
	int user_request_id = request.header.user_request_id;
	int model_id = model_id_seed++;
	load_model_tracking[user_request_id] = 0;
	
	for (int i=0; i< workers.size(); i++) { // loads the model on all the workers before returning a response to the client
	
		int action_id = action_id_seed++;
		
		// Translate clientapi request into a workerapi action
		auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
		load_model->id = action_id;
		load_model->model_path = request.remote_path;

		load_model->model_id = model_id;
		load_model->earliest = 0;
		load_model->latest = util::now() + 100000000000UL; // 10sec slo

		// When the result is received, call this callback
		auto result_callback = [this, callback, user_request_id, model_id] (std::shared_ptr<workerapi::Result> result) {
			std::cout << "Worker  -> " << result->str() << std::endl;

			// Translate workerapi result into a clientapi response
			clientapi::LoadModelFromRemoteDiskResponse response;
			response.header.user_request_id = user_request_id;
			if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
				response.header.status = clockworkError;
				response.header.message = error->message;
			} else if (auto load_model_result = std::dynamic_pointer_cast<workerapi::LoadModelFromDiskResult>(result)) {
				response.header.status = clockworkSuccess;
				response.header.message = "";
				response.model_id = model_id;
				response.input_size = load_model_result->input_size;
				response.output_size = load_model_result->output_size;
			} else {
				response.header.status = clockworkError;
				response.header.message = "Internal Controller Error";
			}

			// Send the response
			std::cout << "Client <-  " << response.str() << std::endl;
			load_model_tracking[user_request_id]++;
			if (load_model_tracking[user_request_id] == workers.size()){
				load_model_tracking.erase(user_request_id);
				callback(response);
			} 
		};
		save_callback(action_id, result_callback);

		// Send the action
		std::cout << "Worker <-  " << load_model->str() << std::endl;
		std::vector<std::shared_ptr<workerapi::Action>> actions;
		actions = {load_model};
		std::cout << ">>> load model: " << model_id << " on worker " << i <<  " <<< \n";
		workers[i]->sendActions(actions);
	}
}