#include "clockwork/workload/workload.h"
#include "clockwork/api/api_common.h"
#include <dmlc/logging.h>
#include "clockwork/util.h"

using namespace clockwork::workload;

void Engine::AddWorkload(Workload* workload) {
	workloads.push_back(workload);
	workload->SetEngine(this);
}

void Engine::SetTimeout(uint64_t timeout, std::function<void(void)> callback) {
	queue.push(element{now + timeout, callback});
}

void Engine::InferComplete(Workload* workload, unsigned model_index) {
	auto callback = [this, workload, model_index]() {
		workload->InferComplete(now, model_index);
	};
	runqueue.push(callback);
}

void Engine::InferError(Workload* workload, unsigned model_index, int status) {
	std::function<void(void)> callback;
	if (status == clockworkInitializing) {
		callback = [this, workload, model_index]() {
			workload->InferErrorInitializing(now, model_index);
		};
	} else {
		callback = [this, workload, model_index, status]() {
			workload->InferError(now, model_index, status);
		};
	}
	runqueue.push(callback);
}

void Engine::Run(clockwork::Client* client) {
	while (true) {
		try {
			auto models = client->ls();
			break;
		} catch (const clockwork_initializing& e1) {
			std::cout << "Clockwork initializing, retrying " << e1.what() << std::endl;
			usleep(1000000);
		} catch (const std::runtime_error& e2) {
			std::cout << "LS error: " << e2.what() << std::endl;
			exit(1);
		}
	}

	now = util::now();
	for (auto &workload : workloads) {
		workload->Start(now);
		running++;
	}
	while (running > 0) {
		// Process all pending results
		now = util::now();
		std::function<void(void)> callback;
		while (runqueue.try_pop(callback)) {
			callback();
		}

		// Run one next request if available
		if (!queue.empty() && queue.top().ready <= now) {
			(queue.top().callback)();
			queue.pop();
		} else {
			usleep(1);
		}
	}
}

Workload::Workload(int id) : user_id(id) {
}

Workload::Workload(int id, clockwork::Model* model) : user_id(id) {
	AddModel(model);
}

Workload::Workload(int id, std::vector<clockwork::Model*> &models) : user_id(id) {
	AddModels(models);
}

void Workload::AddModel(clockwork::Model* model) {
	model->set_user_id(user_id);
	models.push_back(model);
}

void Workload::AddModels(std::vector<clockwork::Model*> &models) {
	for (auto &model : models) {
		AddModel(model);
	}
}

void Workload::SetEngine(Engine* engine) {
	this->engine = engine;
}

void Workload::Infer(unsigned model_index) {
	CHECK(model_index < models.size()) << "Workload " << user_id
		<< " inferring on non-existent model ";
	auto &model = models[model_index];

	std::vector<uint8_t> input(model->input_size());

	auto onSuccess = [this, model_index](std::vector<uint8_t> &output) {
		engine->InferComplete(this, model_index);
	};

	auto onError = [this, model_index](int status, std::string message) {
		engine->InferError(this, model_index, status);
	};

	model->infer(input, onSuccess, onError);
}

void Workload::SetTimeout(uint64_t timeout, std::function<void(void)> callback) {
	engine->SetTimeout(timeout, callback);
}


void Workload::InferErrorInitializing(uint64_t now, unsigned model_index) {
	InferError(now, model_index, clockworkInitializing);
}

ClosedLoop::ClosedLoop(int id, clockwork::Model* model, unsigned concurrency) :
	Workload(id, model), concurrency(concurrency), num_requests(UINT_MAX) {
	CHECK(concurrency != 0) << "ClosedLoop with concurrency 0 created";
	CHECK(num_requests != 0) << "ClosedLoop with num_requests 0 created";
}

ClosedLoop::ClosedLoop(int id, clockwork::Model* model, unsigned concurrency,
	unsigned num_requests) :
	Workload(id, model), concurrency(concurrency), num_requests(num_requests) {
	CHECK(concurrency != 0) << "ClosedLoop with concurrency 0 created";
	CHECK(num_requests != 0) << "ClosedLoop with num_requests 0 created";
}

void ClosedLoop::Start(uint64_t now) {
	for (unsigned i = 0; i < concurrency; i++) {
		Infer();
	}
}

void ClosedLoop::InferComplete(uint64_t now, unsigned model_index) {
	if ((num_requests--) > 0) { Infer(); } else { engine->running--; }
}

void ClosedLoop::InferError(uint64_t now, unsigned model_index, int status) {
	if ((num_requests--) > 0) { Infer(); } else { engine->running--; }
}

void ClosedLoop::InferErrorInitializing(uint64_t now, unsigned model_index) {
	if ((num_requests--) > 0) { Infer(); } else { engine->running--; }
}
