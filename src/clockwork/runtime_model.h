#ifndef _CLOCKWORK_STAGING_H_
#define _CLOCKWORK_STAGING_H_

#include <deque>
#include <unordered_map>

namespace clockwork {

const int clockworkSuccess = 0;
const int clockworkError = 1;

struct RequestHeader {
	// Placeholder
}

struct ResponseHeader {
	int status;
	std::string message;
}

/**
TODO: loading model across network not included
**/

struct LoadModelFromDiskRequest {
	RequestHeader header;
	std::string model_path;
}

struct LoadModelFromDiskResponse {
	ResponseHeader header;
	int model_id;
}

struct InferenceRequest {
	RequestHeader header;
	int model_id;
	int input_size;
	char* input;
	int output_size;
	char* output;
};

struct InferenceResponse {
	ResponseHeader header;
};

/** The opposite of clockwork; manages everything itself */
class SelfDrivingWorker {
public:
	int model_id_seed = 0;
	std::unordered_map<int, ModelManager*> models;

	std::future<LoadModelFromDiskResponse> loadModelFromDisk(LoadModelFromDiskRequest &request) {

	}

	std::future<InferenceResponse> infer(InferenceRequest &request) {
		auto& it = models.find(request.model_id);
		ModelManager* manager;
		if (it == models.end() || (manager = it->second) == nullptr) {
			std::stringstream errorMsg;
			errorMsg << "No model exists with ID " << request.model_id;

			response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
			return response.get_future();
		}
		return manager->add_request(request);
	}
};

/** Manages a specific model instance */
class ModelManager {
public:
	struct Request {
		char* input;
		char* output;
		std::promise<InferenceResponse> promise;
	};

	// The model being managed
	RuntimeModel model;

	std::mutex queue_mutex;
	std::deque<Request> pending_requests;
	std::atomic_flag in_use;	// Only one request can execute at a time for a model

	std::future<InferenceResponse> add_request(InferenceRequest &request) {
		std::promise<InferenceResponse> response;

		if (request.input_size != model.warm->inputsize()) {
			std::stringstream errorMsg;
			errorMsg << "Mismatched input size, expected " << model.warm->inputsize() << ", got " << request.input_size;

			response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
			return response.get_future();
		}
		if (request.output_size != model.warm->outputsize()) {
			std::stringstream errorMsg;
			errorMsg << "Mismatched output size, expected " << model.warm->outputsize() << ", got " << request.output_size;

			response.set_value(InferenceResponse{ResponseHeader{clockworkError, errorMsg.str()}});
			return response.get_future();
		}

		queue_mutex.lock();
		pending_requests.push_back(Request{request.input, request.output, response});
		if (!in_use.test_and_set()) {
			Request toSubmit = pending_requests.pop_front();
			queue_mutex.unlock();
			submit(toSubmit);
		} else {
			queue_mutex.unlock();
		}

		return response.get_future();
	}

private:

	void handle_response(Request request) {
		request.promise.set_value(InferenceResponse{clockworkSuccess, ""});

		queue_mutex.lock();
		if (pending_requests.size() > 0) {
			Request toSubmit = pending_requests.pop_front();
			queue_mutex.unlock();
			submit(toSubmit);
		} else {
			model.unlock();
			in_use.clear();
			queue_mutex.unlock();
		}
	}

	void submit(Request request) {
		RuntimeModel::State state = model.lock();

		RequestBuilder* builder = model.runtime->newRequest();

		if (state == RuntimeModel::State::Warm) {
			builder->addTask(TaskType::PCIe_H2D_Weights, [&model] {
				model.warmToHot();
		    });
		}
		if (state == RuntimeModel::State::Exec) {
	    	builder->addTask(TaskType::PCIe_H2D_Inputs, [&model, &request] {
	    		model.setInput(request.input);
	    	});
		} else {
	    	builder->addTask(TaskType::PCIe_H2D_Inputs, [&model, &request] {
	    		model.hotToExec();
	    		model.setInput(request.input);
	    	});
		}
		builder->addTask(TaskType::GPU, [&model] {
    		model.call();
    	});
    	builder->addTask(TaskType::PCIe_D2H_Output, [&model, &request] {
    		model.getOutput(request.output);
    	});
    	builder->addTask(TaskType::Sync, [this, &request] {
    		this->handle_response(request);
    	});
	}

};

class ParamsEvictionHandler : public EvictionHandler {
private:
	RuntimeModel* model;
public:
	ParamsEvictionHandler(RuntimeModel* model) : model(model) {}

	// evicted is always called while holding the cache lock
	// TODO: unloading hot model isn't cheap (~100 microseconds for cuda module unload)
	//   so don't do it here?
	void evicted() {
		model->hotToWarm();
	}
};

class WorkspaceEvictionHandler : public EvictionHandler {
private:
	RuntimeModel* model;
public:
	WorkspaceEvictionHandler(RuntimeModel* model) : model(model) {}

	// evicted is always called while holding the cache lock
	void evicted() {
		model->execToHot();
	}
};

/** Model is not concurrent, with the exception of the eviction handlers, which may be called
while holding the cache lock */
class RuntimeModel {
public:

	enum State { Warm, Hot, Exec };

private:

	PageCache* cache;

	model::ColdModel* cold;
	model::CoolModel* cool = nullptr;
	model::WarmModel* warm = nullptr;
	model::HotModel* hot = nullptr;
	model::ExecModel* exec = nullptr;

	std::shared_ptr<Allocation> params_pages = nullptr;
	std::shared_ptr<Allocation> workspace_pages = nullptr;

	EvictionCallback* params_callback = nullptr;
	EvictionCallback* workspace_callback = nullptr;

public:

	RuntimeModel(PageCache* cache, model::ColdModel* cold) : cache(cache), cold(cold) {
		params_callback = new ParamsEvictionHandler(this);
		workspace_callback = new WorkspaceEvictionHandler(this);
	}

	State lock() {
		if (!cache->try_lock(params_pages)) return State::Warm;
		if (!cache->try_lock(workspace_pages)) return State::Hot;
		return State::Exec;
	}

	void coldToCool() {
		CHECK(cold != nullptr) << "Cannot transition cold -> cool, cold == nullptr";
		CHECK(cool == nullptr) << "Cannot transition cold -> cool, cool already exists";
		cool = cold->load();
	}

	void coolToWarm() {
		CHECK(cool != nullptr) << "Cannot transition cool -> warm, cool == nullptr";
		CHECK(warm == nullptr) << "Cannot transition cool -> warm, cool already exists";
		warm = cool->load();
	}

	void warmToHot() {
		CHECK(warm != nullptr) << "Cannot transition warm -> hot, warm == nullptr";
		CHECK(hot == nullptr) << "Cannot transition warm -> hot, hot != nullptr";
		CHECK(params_pages == nullptr) << "Cannot transition warm -> hot, params_pages already allocated";
		params_pages = cache->alloc(warm->num_params_pages(runtime->cache->page_size), params_callback);
		hot = warm->load(params_pages);
	}

	void hotToExec() {
		CHECK(hot != nullptr) << "Cannot transition hot -> exec, hot == nullptr";
		CHECK(exec == nullptr) << "Cannot transition hot -> exec, exec != nullptr";
		CHECK(workspace_pages == nullptr) << "Cannot transition hot -> exec, workspace_pages already allocated";
		workspace_pages = cache->alloc(hot->num_workspace_pages(runtime->cache->page_size), workspace_callback);
		exec = hot->load(workspace_pages);
	}

	void setInput(void* input) {
		CHECK(exec != nullptr) << "Cannot set input on exec == nullptr";
		exec->set_input(input);
	}

	void call() {
		CHECK(exec != nullptr) << "Cannot call exec == nullptr";
		exec->call();
	}

	void getOutput(void* output) {
		CHECK(exec != nullptr) << "Cannot get output of exec == nullptr";
		exec->get_output(output);
	}

	void execToHot() {
		CHECK(exec != nullptr) << "Cannot transition exec -> hot, exec == nullptr";
		CHECK(workspace_pages != nullptr) << "Cannot transition exec -> hot, workspace_pages == nullptr";
		workspace_pages = nullptr;
		exec->unload();
		exec = nullptr;
	}

	void hotToWarm() {
		CHECK(hot != nullptr) << "Cannot transition hot -> warm, hot == nullptr";
		CHECK(params_pages != nullptr) << "Cannot transition hot -> warm, params_pages == nullptr";
		params_pages = nullptr;
		exec->unload();
		exec = nullptr;
	}

	void warmToCool() {
		CHECK(warm != nullptr) << "Cannot transition warm -> cool, warm == nullptr";
		warm->unload();
		warm = nullptr;
	}

	void coolToCold() {
		CHECK(cool != nullptr) << "Cannot transition cool -> cold, cool == nullptr";
		cool->unload();
		cool = nullptr;
	}

	void unlock() {
		cache->unlock(workspace_pages);
		cache->unlock(params_pages);
	}

}

}

#endif