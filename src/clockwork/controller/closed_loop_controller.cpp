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
#include "closed_loop_controller.h"


using namespace clockwork;

ClosedLoopControllerImpl::ClosedLoopControllerImpl(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs, int max_batch_size) : Controller::Controller(client_port, worker_host_port_pairs), model_id_seed(0), action_id_seed(0), max_batch_size(max_batch_size), batch_size(1) {

	ClosedLoopControllerImpl::load_example_model();
}
	
void ClosedLoopControllerImpl::save_callback(int id, std::function<void(std::shared_ptr<workerapi::Result>)> callback) {
		std::lock_guard<std::mutex> lock(actions_mutex);

		auto it = action_callbacks.find(id);
		CHECK(it == action_callbacks.end()) << "ID " << id << " already exists";

		action_callbacks.insert(std::make_pair(id, callback));
}

std::function<void(std::shared_ptr<workerapi::Result>)> ClosedLoopControllerImpl::get_callback(int id) {
		std::lock_guard<std::mutex> lock(ClosedLoopControllerImpl::actions_mutex);

		auto it = ClosedLoopControllerImpl::action_callbacks.find(id);

		if (it == action_callbacks.end()) return nullptr;

		auto callback = it->second;
		ClosedLoopControllerImpl::action_callbacks.erase(it);
		return callback;
}

// workerapi::Controller::sendResult
void  ClosedLoopControllerImpl::sendResult(std::shared_ptr<workerapi::Result> result) {
		auto callback = get_callback(result->id);
		CHECK(callback != nullptr) << "Received result without a corresponding action: " << result->str();
		callback(result);
}

void ClosedLoopControllerImpl::closedLoopController(){

		std::vector<std::shared_ptr<workerapi::Action>> actions;
		void* example_model_input = malloc(ClosedLoopControllerImpl::example_model_input_size);
		char* input = (char*)malloc(sizeof(char) * ClosedLoopControllerImpl::example_model_input_size * ClosedLoopControllerImpl::batch_size);
		for (int i = 0; i < ClosedLoopControllerImpl::batch_size; i++) {
			memcpy(input + (i * ClosedLoopControllerImpl::example_model_input_size), static_cast<char*>(example_model_input), ClosedLoopControllerImpl::example_model_input_size);
		}
		auto infer = std::make_shared<workerapi::Infer>();
		infer->id = action_id_seed++;
		infer->model_id = 0;
		infer->gpu_id = 0;
		infer->batch_size = ClosedLoopControllerImpl::batch_size;
		infer->input_size = ClosedLoopControllerImpl::example_model_input_size * ClosedLoopControllerImpl::batch_size;
		infer->input = input;
		infer->earliest = weights_available_at[std::make_pair(0, 0)];
		infer->latest = util::now() + 1000000000UL;

		auto infer_complete = [this, input, example_model_input] (std::shared_ptr<workerapi::Result> result) {

			std::cout << "Worker  -> " << result->str() << std::endl;
			free(example_model_input);
			free(input);
			batch_size = std::max((batch_size * 2 ) % (max_batch_size * 2), 1); 
			closedLoopController();
		};
		ClosedLoopControllerImpl::save_callback(infer->id, infer_complete);
		actions.push_back(infer);
		ClosedLoopControllerImpl::workers[0]->sendActions(actions);
		std::cout << "Worker <-  " << infer->str() << " batch_size=" << batch_size << std::endl;
}

void ClosedLoopControllerImpl::load_weights_example_model(int gpu_id, int model_id) {
		auto load_weights = std::make_shared<workerapi::LoadWeights>();
		load_weights->model_id = model_id;
		load_weights->gpu_id = gpu_id;
		load_weights->earliest = util::now();
		load_weights->latest = load_weights->earliest + 10000000000L;

		auto load_weights_complete = [this,  model_id, gpu_id] (std::shared_ptr<workerapi::Result> result) {
			std::cout << "Worker  -> " << result->str() << std::endl;

			// If the result wasn't a success, then mark the weights as unavailble again
			if (auto error_result = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
				weights_available_at.erase(std::make_pair(model_id, gpu_id));
			}
			
			closedLoopController();

		};

		// Store when the weights will be available
		weights_available_at[std::make_pair(model_id, gpu_id)] = util::now() + 7000000UL; // Assume it'll take 7 ms to load weights
		std::vector<std::shared_ptr<workerapi::Action>> actions;
		actions = {load_weights};
		for (int i = 0; i < workers.size(); i++){
			load_weights->id = action_id_seed++;
			ClosedLoopControllerImpl::save_callback(load_weights->id, load_weights_complete);
			std::cout << "Worker " << i << " <-  " << load_weights->str() << std::endl;
			ClosedLoopControllerImpl::workers[i]->sendActions(actions);
		}
		actions.clear();
}

void ClosedLoopControllerImpl::load_example_model() {

		int model_id = ClosedLoopControllerImpl::model_id_seed++;
		auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
		load_model->model_id = model_id;
		load_model->no_of_copies = 1;
		load_model->model_path = util::get_example_model_path();
		load_model->earliest = 0;
		load_model->latest = util::now() + 100000000000UL;
		// When the result is received, call this callback
		auto result_callback =  [this](std::shared_ptr<workerapi::Result> result) {
			std::cout << "Worker  -> " << result->str() << std::endl;
			auto load_model_result = std::dynamic_pointer_cast<workerapi::LoadModelFromDiskResult>(result);
			example_model_input_size = load_model_result->input_size;
			load_weights_example_model(0, 0);
		};
		std::vector<std::shared_ptr<workerapi::Action>> actions;
		actions = {load_model};
	
	for (int i = 0; i < workers.size(); i++) {
			load_model->id =  action_id_seed++;
			ClosedLoopControllerImpl::save_callback(load_model->id, result_callback);
			std::cout << "Worker " << i <<" <-  " << load_model->str() << std::endl;
			ClosedLoopControllerImpl::workers[i]->sendActions(actions);
		}
		actions.clear();
}

void ClosedLoopControllerImpl::uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
		std::cout << "Client  -> " << request.str() << std::endl;
		clientapi::UploadModelResponse response;
		std::cout << "Client <-  " << response.str() << std::endl;
		callback(response);
}

void ClosedLoopControllerImpl::infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {}
	
/** This is a 'backdoor' API function for ease of experimentation */
void ClosedLoopControllerImpl::evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
		std::cout << "Client  -> " << request.str() << std::endl;
		clientapi::EvictResponse response;
		std::cout << "Client <-  " << response.str() << std::endl;
		callback(response);
}

/** This is a 'backdoor' API function for ease of experimentation */
void ClosedLoopControllerImpl::loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
		std::cout << "Client  -> " << request.str() << std::endl;

		int user_request_id = request.header.user_request_id;
		int action_id = action_id_seed++;
		int model_id = model_id_seed++;
		// Translate clientapi request into a workerapi action
		auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
		load_model->id = action_id;
		load_model->model_id = model_id;
		load_model->model_path = util::get_example_model_path();
		load_model->no_of_copies = 1;
		load_model->earliest = 0;
		load_model->latest = util::now() + 10000000000UL;

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
			callback(response);
		};
		ClosedLoopControllerImpl::save_callback(action_id, result_callback);

		// Send the action
		std::cout << "Worker <-  " << load_model->str() << std::endl;
		std::vector<std::shared_ptr<workerapi::Action>> actions;
		actions = {load_model};
		ClosedLoopControllerImpl::workers[0]->sendActions(actions);
}

