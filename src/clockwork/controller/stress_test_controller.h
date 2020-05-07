#ifndef _CLOCKWORK_CONTROLLER_STRESS_TEST_CONTROLLER_H_
#define _CLOCKWORK_CONTROLLER_STRESS_TEST_CONTROLLER_H_

#include "clockwork/network/controller.h"
#include "clockwork/api/worker_api.h"
#include "controller.h"
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unistd.h>
#include <sys/types.h>

using namespace clockwork;

// Simple controller implementation that send requests to worker and wait for response
class StressTestController : public Controller {
public:
	std::string model_path = "/home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model";
	unsigned duplicates = 100;
	unsigned max_models_on_gpu = 50;
	size_t input_size = 602112;
	char* input;

	std::atomic_int action_id_seed;

	std::mutex mutex;
	unsigned outstanding_loadweights = 0;
	unsigned outstanding_infer = 0;
	std::deque<unsigned> models_on_gpu;
	std::deque<unsigned> models_not_on_gpu;

	unsigned load_weights_errors = 0;
	unsigned infer_errors = 0;
	uint64_t profiled_load_weights;
	uint64_t profiled_exec;
	std::vector<uint64_t> load_weights_measurements;
	std::vector<uint64_t> infer_measurements;

	std::thread printer;

	std::map<unsigned, std::function<void(std::shared_ptr<workerapi::Result>)>> callbacks;

	StressTestController(int client_port, std::vector<std::pair<std::string, std::string>> worker_host_port_pairs):
		Controller::Controller(client_port, worker_host_port_pairs), action_id_seed(0), printer(&StressTestController::printerThread, this) {
		input = static_cast<char*>(malloc(input_size));

		init();
	}

	std::string stats(std::vector<uint64_t> v, uint64_t profiled, uint64_t duration) {
		if (v.size() == 0) {
			return "throughput=0";
		}

		const auto [min, max] = std::minmax_element(v.begin(), v.end());
		int count = v.size();
		double sum = std::accumulate(v.begin(), v.end(), 0.0);
		double throughput = count * 1000000000.0 / static_cast<double>(duration);

		std::stringstream s;
		s << std::fixed << std::setprecision(2);
		s << "profiled=" << profiled << " min=" << *min << " max=" << *max << " mean=" << (sum/count) << " throughput=" << throughput << " efficiency=" << (sum/((float) duration));
		return s.str();
	}

	void printerThread() {
		uint64_t last_print = util::now();
		uint64_t print_interval = 1000000000UL;

		while (true) {
			if (last_print + print_interval > util::now()) {
				usleep(10000);
				continue;
			}

			std::vector<uint64_t> load_weights;
			std::vector<uint64_t> infer;
			unsigned infer_errors;
			unsigned load_weights_errors;
			uint64_t now;
			{
				std::lock_guard<std::mutex> lock(mutex);

				now = util::now();

				load_weights = this->load_weights_measurements;
				infer = this->infer_measurements;
				infer_errors = this->infer_errors;
				load_weights_errors = this->load_weights_errors;

				this->load_weights_measurements.clear();
				this->infer_measurements.clear();
				infer_errors = 0;
				load_weights_errors = 0;
			}
			uint64_t duration = now - last_print;
			last_print = now;

			std::stringstream report;
			report << "LoadWeights errors=" << load_weights_errors << " " << stats(load_weights, profiled_load_weights, duration) << std::endl;
			report << "Infer errors=" << infer_errors << " " << stats(infer, profiled_exec, duration);

			std::cout << report.str() << std::endl;
		}
	}
	
	void save_callback(unsigned action_id, std::function<void(std::shared_ptr<workerapi::Result>)> callback) {
		auto it = callbacks.find(action_id);
		CHECK(it == callbacks.end()) << "Action " << action_id << " already exists";
		callbacks.insert(std::make_pair(action_id, callback));
	}

	void callback(std::shared_ptr<workerapi::Result> result) {
		auto it = callbacks.find(result->id);
		CHECK(it != callbacks.end()) << "Received result with no callback";

		auto f = it->second;
		callbacks.erase(it);

		f(result);
	}

	void init() {
		std::lock_guard<std::mutex> lock(mutex);

		unsigned action_id = action_id_seed++;

		auto load_model = std::make_shared<workerapi::LoadModelFromDisk>();
		load_model->id = action_id;
		load_model->model_id = 0;
		load_model->model_path = model_path;
		load_model->earliest = 0;
		load_model->latest = util::now() + 100000000000UL;
		load_model->no_of_copies = duplicates;

		save_callback(action_id, std::bind(&StressTestController::onLoadModelsComplete, this, std::placeholders::_1));

		this->workers[0]->sendAction(load_model);
	}

	void onLoadModelsComplete(std::shared_ptr<workerapi::Result> result) {
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			CHECK(false) << "StressTestController unable to load initial models " << result->str();
		} else if (auto load = std::dynamic_pointer_cast<workerapi::LoadModelFromDiskResult>(result)) {
			profiled_load_weights = load->weights_load_time_nanos;
			profiled_exec = load->batch_size_exec_times_nanos[0];
			for (unsigned i = 0; i < duplicates; i++) {
				models_not_on_gpu.push_back(i);
			}
		} else {
			CHECK(false) << "Unexpected response to EvictWeights action";
		}
	}

	bool evictNext() {
		if (models_on_gpu.size() <= max_models_on_gpu) return false;

		unsigned model_id = models_on_gpu.front();
		models_on_gpu.pop_front();

		unsigned action_id = action_id_seed++;

		auto evict = std::make_shared<workerapi::EvictWeights>();
		evict->id = action_id;
		evict->model_id = model_id;
		evict->gpu_id = 0;
		evict->earliest = util::now() - 10000000UL; // 10 ms ago
		evict->latest = evict->earliest + 10000000000UL; // 10s

		save_callback(action_id, std::bind(&StressTestController::onEvictWeightsComplete, this, model_id, std::placeholders::_1));

		this->workers[0]->sendAction(evict);

		return true;
	}

	void onEvictWeightsComplete(unsigned model_id, std::shared_ptr<workerapi::Result> result) {
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			CHECK(false) << "Error in evict weights action, which should never happen except for fatal errors";
		} else if (auto evict = std::dynamic_pointer_cast<workerapi::EvictWeightsResult>(result)) {
			models_not_on_gpu.push_back(model_id);
		} else {
			CHECK(false) << "Unexpected response to EvictWeights action";
		}
	}

	bool loadNext() {
		if (models_not_on_gpu.size() == 0) return false;
		if (outstanding_loadweights >= 4) return false;

		unsigned model_id = models_not_on_gpu.front();
		models_not_on_gpu.pop_front();

		unsigned action_id = action_id_seed++;

		auto load = std::make_shared<workerapi::LoadWeights>();
		load->id = action_id;
		load->gpu_id = 0;
		load->model_id = model_id;
		load->earliest = util::now() - 10000000UL; // 10 ms ago
		load->latest = load->earliest + 10000000000UL; // 10s

		save_callback(action_id, std::bind(&StressTestController::onLoadWeightsComplete, this, model_id, std::placeholders::_1));

		this->workers[0]->sendAction(load);

		outstanding_loadweights++;

		return true;
	}

	void onLoadWeightsComplete(unsigned model_id, std::shared_ptr<workerapi::Result> result) {
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			std::cout << "LoadWeights error " << result->str() << std::endl;
			load_weights_errors++;
			models_not_on_gpu.push_back(model_id);
		} else if (auto load = std::dynamic_pointer_cast<workerapi::LoadWeightsResult>(result)) {
			models_on_gpu.push_back(model_id);
			load_weights_measurements.push_back(load->duration);
		} else {
			CHECK(false) << "Unexpected response to LoadWeights action";
		}
		outstanding_loadweights--;
	}

	bool inferNext() {
		if (outstanding_infer >= 4) return false;

		unsigned next_model_ix = 6 + outstanding_infer;
		if (models_on_gpu.size() <= next_model_ix) return false;

		unsigned model_id = models_on_gpu[next_model_ix++];
		unsigned action_id = action_id_seed++;

		auto infer = std::make_shared<workerapi::Infer>();
		infer->id = action_id;
		infer->model_id = model_id;
		infer->gpu_id = 0;
		infer->batch_size = 1;
		infer->input = input;
		infer->input_size = input_size;
		infer->earliest = util::now() - 10000000UL; // 10 ms ago
		infer->latest = infer->earliest + 10000000000UL; // 10s

		save_callback(action_id, std::bind(&StressTestController::onInferComplete, this, model_id, std::placeholders::_1));

		this->workers[0]->sendAction(infer);

		outstanding_infer++;

		return true;
	}

	void onInferComplete(unsigned model_id, std::shared_ptr<workerapi::Result> result) {
		if (auto error = std::dynamic_pointer_cast<workerapi::ErrorResult>(result)) {
			std::cout << "Infer error " << result->str() << std::endl;
			infer_errors++;
		} else if (auto infer = std::dynamic_pointer_cast<workerapi::InferResult>(result)) {
			infer_measurements.push_back(infer->exec.duration);
		} else {
			CHECK(false) << "Unexpected response to LoadWeights action";
		}
		outstanding_infer--;
	}

	void sendNewActions() {
		while (evictNext());
		while (loadNext());
		while (inferNext());
	}

	virtual void sendResult(std::shared_ptr<workerapi::Result> result) {
		std::lock_guard<std::mutex> lock(mutex);

		callback(result);
		sendNewActions();
	}

	virtual void infer(clientapi::InferenceRequest &request, std::function<void(clientapi::InferenceResponse&)> callback) {
		CHECK(false) << "infer from client not supported";
	}
	virtual void uploadModel(clientapi::UploadModelRequest &request, std::function<void(clientapi::UploadModelResponse&)> callback) {
		CHECK(false) << "uploadModel not supported";
	}
	virtual void evict(clientapi::EvictRequest &request, std::function<void(clientapi::EvictResponse&)> callback) {
		CHECK(false) << "evict not supported";
	}
	virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request, std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback) {
		CHECK(false) << "loadRemoteModel not supported";
	}
};

#endif