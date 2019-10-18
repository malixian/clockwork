#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/task.h"
#include "clockwork/action.h"
#include <catch2/catch.hpp>
#include "clockwork/test/actions.h"

using namespace clockwork;
using namespace clockwork::model;

class TestAction {
public:
    std::atomic_bool is_success = false;
    std::atomic_bool is_error = false;
    int status_code;
    std::string error_message;

    void setsuccess() {
        is_success = true;
    }

    void seterror(std::shared_ptr<workerapi::ErrorResult> result) {
        is_error = true;
        this->status_code = result->status;
        this->error_message = result->message;
    }

    void await() {
        while ((!is_success) && (!is_error));
    }

    void check_success(bool expect_success, int expected_status_code = 0) {
        if (expect_success) {
            INFO(status_code << ": " << error_message);
            REQUIRE(!is_error);
            REQUIRE(is_success);
        } else {
            REQUIRE(is_error);
            REQUIRE(!is_success);
            INFO(status_code << ": " << error_message);
            REQUIRE(status_code == expected_status_code);
        }
    }
};

class TestLoadModelFromDiskAction : public LoadModelFromDiskAction, public TestAction {
public:
    TestLoadModelFromDiskAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadModelFromDisk> action) : 
        LoadModelFromDiskAction(runtime, action) {}

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
        setsuccess();
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        seterror(result);
    }

};

class TestLoadWeightsAction : public LoadWeightsAction, public TestAction {
public:
    TestLoadWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::LoadWeights> action) : 
        LoadWeightsAction(runtime, action) {}

    void success(std::shared_ptr<workerapi::LoadWeightsResult> result) {
        setsuccess();
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        seterror(result);
    }

};

class TestEvictWeightsAction : public EvictWeightsAction, public TestAction {
public:
    TestEvictWeightsAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::EvictWeights> action) : 
        EvictWeightsAction(runtime, action) {}

    void success(std::shared_ptr<workerapi::EvictWeightsResult> result) {
        setsuccess();
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        seterror(result);
    }

};

class TestInferAction : public InferAction, public TestAction {
public:
    TestInferAction(ClockworkRuntime* runtime, std::shared_ptr<workerapi::Infer> action) : 
        InferAction(runtime, action) {}

    void success(std::shared_ptr<workerapi::InferResult> result) {
        setsuccess();
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        seterror(result);
    }

};

class ClockworkRuntimeWrapper : public ClockworkRuntime {
public:
    ~ClockworkRuntimeWrapper() {
        this->shutdown(true);
    }
};

std::shared_ptr<ClockworkRuntime> make_runtime() {
    return std::make_shared<ClockworkRuntimeWrapper>();
}

Model* make_model_for_action() {
    std::string f = clockwork::util::get_example_model();

    Model* model = Model::loadFromDisk(f+".so", f+".clockwork", f+".clockwork_params");
    model->instantiate_model_on_host();
    model->instantiate_model_on_device();
    return model;
}

TEST_CASE("Load Model From Disk Action", "[action] [loadmodel_action]") {
    auto clockwork = make_runtime();

    auto load_model = new TestLoadModelFromDiskAction(clockwork.get(), load_model_from_disk_action());

    load_model->submit();
    load_model->await();
    load_model->check_success(true);

    delete load_model;
    
}

TEST_CASE("Load Model From Disk Action Multiple", "[action] [loadmodel_action]") {
    auto clockwork = make_runtime();

    for (unsigned i = 0; i < 10; i++) {
        auto action = load_model_from_disk_action();
        action->model_id = i;

        auto load_model = new TestLoadModelFromDiskAction(clockwork.get(), action);

        load_model->submit();
        load_model->await();
        load_model->check_success(true);

        delete load_model;
    }

    
}

TEST_CASE("Load Weights Action", "[action] [loadweights_action]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;
    
}

TEST_CASE("Load Weights Action Multiple", "[action] [loadweights_multiple]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    for (unsigned i = 0; i < 5; i++) {
        auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

        load_weights->submit();
        load_weights->await();
        load_weights->check_success(true);

        delete load_weights;
    }

    
}

TEST_CASE("Load Weights Action Invalid Model", "[action]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(false, actionErrorUnknownModel);

    delete load_weights;
    
}

TEST_CASE("Load Evict Weights Action", "[action] [evict_action]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    auto evict_weights = new TestEvictWeightsAction(clockwork.get(), evict_weights_action());

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(true);

    delete load_weights;
    delete evict_weights;
    
}

TEST_CASE("Evict without Weights Action", "[action] [evict_action]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    auto evict_weights = new TestEvictWeightsAction(clockwork.get(), evict_weights_action());

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(false, actionErrorModelWeightsNotPresent);

    delete evict_weights;
    
}

TEST_CASE("Infer Action", "[action] [infer_action]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    auto infer = new TestInferAction(clockwork.get(), infer_action(model));

    infer->submit();
    infer->await();
    infer->check_success(true);

    delete load_weights;
    delete infer;
    
}

TEST_CASE("Infer Action Wrong Input Size", "[action] [nomodel]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();

    auto action = infer_action();
    action->input_size = 10;
    auto infer = new TestInferAction(clockwork.get(), action);

    infer->submit();
    infer->await();
    infer->check_success(false, actionErrorUnknownModel);

    delete infer;
    
}

TEST_CASE("Infer Action Nonexistent Model", "[action] [nomodel]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();

    auto infer = new TestInferAction(clockwork.get(), infer_action());

    infer->submit();
    infer->await();
    infer->check_success(false, actionErrorUnknownModel);

    delete infer;
    
}

TEST_CASE("Infer Action Nonexistent Weights", "[action] [noweights]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();

    auto load_model = new TestLoadModelFromDiskAction(clockwork.get(), load_model_from_disk_action());

    load_model->submit();
    load_model->await();
    load_model->check_success(true);

    delete load_model;

    auto infer = new TestInferAction(clockwork.get(), infer_action());

    infer->submit();
    infer->await();
    infer->check_success(false, actionErrorModelWeightsNotPresent);

    delete infer;
    
}

TEST_CASE("Infer Action Multiple", "[action] [infer_action_multiple]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;

    for (unsigned i = 0; i < 2; i++) {
        auto infer = new TestInferAction(clockwork.get(), infer_action(model));

        infer->submit();
        infer->await();
        infer->check_success(true);

        delete infer;
    }

    
}

TEST_CASE("Make Many Models", "[action] [models]") {
    std::vector<Model*> models;
    for (unsigned i = 0; i < 30; i++) {
        models.push_back(make_model_for_action());
    }
    for (Model* model : models) {
        delete model;
    }
}

TEST_CASE("Infer Action Multiple Concurrent", "[action] [infer_action_concurrent]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;

    std::vector<TestInferAction*> infers;

    for (unsigned i = 0; i < 2; i++) {
        infers.push_back(new TestInferAction(clockwork.get(), infer_action(model)));
    }

    for (TestInferAction* infer : infers) {
        infer->submit();
    }

    for (TestInferAction* infer : infers) {
        infer->await();
        infer->check_success(true);

        delete infer;
    }

    
}

TEST_CASE("Infer after Evict Action", "[action] [evict_action]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;
    
    auto infer = new TestInferAction(clockwork.get(), infer_action(model));

    infer->submit();
    infer->await();
    infer->check_success(true);

    delete infer;

    auto evict_weights = new TestEvictWeightsAction(clockwork.get(), evict_weights_action());

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(true);

    delete evict_weights;

    auto infer2 = new TestInferAction(clockwork.get(), infer_action(model));

    infer2->submit();
    infer2->await();
    infer2->check_success(false, actionErrorModelWeightsNotPresent);

    delete infer2;

    
}

TEST_CASE("Actions E2E", "[action] [e2e]") {
    auto clockwork = make_runtime();

    auto load_model = new TestLoadModelFromDiskAction(clockwork.get(), load_model_from_disk_action());

    load_model->submit();
    load_model->await();
    load_model->check_success(true);

    delete load_model;

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;
    
    Model* model = clockwork->manager->models->get(0)->model;
    auto infer = new TestInferAction(clockwork.get(), infer_action(model));

    infer->submit();
    infer->await();
    infer->check_success(true);

    delete infer;

    auto evict_weights = new TestEvictWeightsAction(clockwork.get(), evict_weights_action());

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(true);

    delete evict_weights;

    auto infer2 = new TestInferAction(clockwork.get(), infer_action(model));

    infer2->submit();
    infer2->await();
    infer2->check_success(false, actionErrorModelWeightsNotPresent);

    delete infer2;

    
}

TEST_CASE("Task Cancelled After Shutdown", "[action] [shutdown]") {
    Model* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    auto action = load_weights_action();
    action->earliest = util::now() + 1000000000UL; // + 10seconds
    action->latest = action->earliest;

    auto load_weights = new TestLoadWeightsAction(clockwork.get(), action);

    load_weights->submit();

    clockwork = nullptr; // destroys the runtime

    load_weights->await();
    load_weights->check_success(false, actionCancelled);

    delete load_weights;
}

class TestLoadModelFromDiskActionThatDeletesItself : public LoadModelFromDiskAction {
public:
    TestAction &action_status;

    TestLoadModelFromDiskActionThatDeletesItself(
            ClockworkRuntime* runtime, 
            std::shared_ptr<workerapi::LoadModelFromDisk> action, 
            TestAction &action_status) : 
        LoadModelFromDiskAction(runtime, action), action_status(action_status) {}

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
        action_status.setsuccess();
        delete this;
    }

    void error(std::shared_ptr<workerapi::ErrorResult> result) {
        action_status.seterror(result);
        delete this;
    }

};

TEST_CASE("Task Action That Deletes Itself in Callback", "[action] [shutdown]") {
    auto clockwork = make_runtime();

    TestAction action_status;

    auto action = load_model_from_disk_action();
    action->model_path = "";
    auto load_model = new TestLoadModelFromDiskActionThatDeletesItself(
            clockwork.get(), action, action_status);

    load_model->submit();
    action_status.await();
    action_status.check_success(false, actionErrorInvalidModelPath);

    
}