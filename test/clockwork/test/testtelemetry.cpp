#include "telemetry.h"

using namespace clockwork;
using namespace clockwork::model;

BatchedModel* create_model_for_action() {
	std::string f = clockwork::util::get_example_model();

	Model* model = Model::loadFromDisk(f+".1.so", f+".1.clockwork", f+".clockwork_params", GPU_ID_0);

	std::vector<std::pair<unsigned, Model*>> models = {{1, model}};
	BatchedModel* batched = new BatchedModel(model->weights_size, model->weights_pinned_host_memory, models, GPU_ID_0);

	batched->instantiate_models_on_host();
	batched->instantiate_models_on_device();
	return batched;
}

bool telemetry_in_range(uint64_t start, uint64_t end, bool infer) {
	bool in_range = true;

	in_range &= telemetry_enqueued > start && telemetry_enqueued < end;
	in_range &= telemetry_dequeued > start && telemetry_dequeued < end;
	if (infer) {
		in_range &= telemetry_exec_complete > start && telemetry_exec_complete < end;
		in_range &= telemetry_async_complete > start && telemetry_async_complete < end;
	}
	return in_range;
}

TEST_CASE("Task Telemetry", "[task_telemetry]"){

    BatchedModel* model = create_model_for_action();
	auto runtime  = std::make_shared<ClockworkRuntimeWrapper>();
	runtime->task_telemetry_logger = new TestTaskTelemetryLogger();

    runtime->manager->models->put(0, GPU_ID_0, new RuntimeModel(model, GPU_ID_0));

	uint64_t start = util::now();

    TestLoadWeightsAction load_weights(runtime.get(), load_weights_action());
    load_weights.submit();
    load_weights.await();

	uint64_t end = util::now();

	load_weights.check_success(true);

	assert(tasks_logged == 1);
	assert(action_id == 0 && model_id == 0);
	assert(telemetry_in_range(start, end, false));

	start = util::now();

	TestInferAction infer(runtime.get(), infer_action(1, model));
    infer.submit();
    infer.await();

	end = util::now();

	infer.check_success(true);

	assert(tasks_logged == 4);
	assert(action_id == 1 && model_id == 0 & batch_size == 1);
	assert(telemetry_in_range(start, end, true));

	start = util::now();

	TestEvictWeightsAction evict_weights(runtime.get(), evict_weights_action());
    evict_weights.submit();
    evict_weights.await();

	end = util::now();

	evict_weights.check_success(true);

	assert(tasks_logged == 5);
	assert(action_id == 2 && model_id == 0);
	assert(telemetry_in_range(start, end, false));

}

TEST_CASE("Action Telemetry", "[action_telemetry]"){
	ClockworkWorker* worker = new ClockworkWorker();
    TestController* controller = new TestController();
	worker->controller = controller;

	uint64_t start = util::now();
	worker->runtime->action_telemetry_logger = new TestActionTelemetryLogger();
	auto load_model = load_model_from_disk_action();

	std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};

	worker->sendActions(actions);
	controller->expect(actionSuccess);
	uint64_t end = util::now();

	assert(actions_logged == 2 && action_id == 0);
	assert(action_timestamp > start && action_timestamp < end);
	
	start = util::now();
	auto load_weights = load_weights_action();
	actions = {load_weights};
	worker->sendActions(actions);
	controller->expect(actionSuccess);
	end = util::now();

	assert(actions_logged == 4 && action_id == 1);
	assert(action_timestamp > start && action_timestamp < end);

	start = util::now();
	auto infer = infer_action2(worker);
	actions = {infer};
	worker->sendActions(actions);
	controller->expect(actionSuccess);
	end = util::now();

	assert(actions_logged == 6 && action_id == 2);
	assert(action_timestamp > start && action_timestamp < end);

	start = util::now();
	auto evict_weights = evict_weights_action();
	actions = {evict_weights};
	worker->sendActions(actions);
	controller->expect(actionSuccess);
	end = util::now();

	assert(actions_logged == 8 && action_id == 3);
	assert(action_timestamp > start && action_timestamp < end);

	worker->shutdown(true);
	delete worker;
}
