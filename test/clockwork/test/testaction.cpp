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

    void seterror(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
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

class TestLoadWeightsAction : public LoadWeightsAction, public TestAction {
public:
    TestLoadWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest) : 
        LoadWeightsAction(runtime, model_id, earliest, latest) {}

    void success() {
        setsuccess();
    }

    void error(int status_code, std::string message) {
        seterror(status_code, message);
    }

};

class TestEvictWeightsAction : public EvictWeightsAction, public TestAction {
public:
    TestEvictWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest) : 
        EvictWeightsAction(runtime, model_id, earliest, latest) {}

    void success() {
        setsuccess();
    }

    void error(int status_code, std::string message) {
        seterror(status_code, message);
    }

};

class TestInferAction : public InferAction, public TestAction {
public:
    TestInferAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest, char* input, char* output) : 
        InferAction(runtime, model_id, earliest, latest, input, output) {}

    void success() {
        setsuccess();
    }

    void error(int status_code, std::string message) {
        seterror(status_code, message);
    }

};

ClockworkRuntime* make_runtime() {
    return new ClockworkRuntime();
}

Model* make_model_for_action() {
    std::string f = clockwork::util::get_example_model();

    Model* model = Model::loadFromDisk(f+".so", f+".clockwork", f+".clockwork_params");
    model->instantiate_model_on_host();
    model->instantiate_model_on_device();
    return model;
}

TEST_CASE("Load Weights Action", "[action] [loadweights_action]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    TestLoadWeightsAction* action = 
        new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    action->submit();

    while ((!action->is_success) && (!action->is_error));

    INFO(action->status_code << ": " << action->error_message);
    REQUIRE(!action->is_error);
    REQUIRE(action->is_success);
}

TEST_CASE("Load Weights Action Multiple", "[action] [loadweights_multiple]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    for (unsigned i = 0; i < 5; i++) {
        TestLoadWeightsAction* action = 
            new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

        action->submit();
        action->await();
        action->check_success(true);

        delete action;
    }
}

TEST_CASE("Load Weights Action Invalid Model", "[action]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();

    TestLoadWeightsAction* action = 
        new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    action->submit();

    while ((!action->is_success) && (!action->is_error));

    REQUIRE(!action->is_success);
    REQUIRE(action->is_error);
    REQUIRE(action->status_code == actionErrorUnknownModel);
}

TEST_CASE("Load Evict Weights Action", "[action] [evict_action]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    TestLoadWeightsAction* load_weights = 
        new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    TestEvictWeightsAction* evict_weights = 
        new TestEvictWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(true);
}

TEST_CASE("Evict without Weights Action", "[action] [evict_action]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    TestEvictWeightsAction* evict_weights = 
        new TestEvictWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(false, actionErrorModelWeightsNotPresent);
}

TEST_CASE("Infer Action", "[action] [infer_action]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    TestLoadWeightsAction* load_weights = 
        new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    char* input = static_cast<char*>(malloc(model->input_size()));
    char* output = static_cast<char*>(malloc(model->output_size()));

    TestInferAction* infer = 
        new TestInferAction(clockwork, 0, util::now(), util::now()+1000000000, input, output);

    infer->submit();
    infer->await();
    infer->check_success(true);
}

TEST_CASE("Infer Action Multiple", "[action] [infer_action]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    TestLoadWeightsAction* load_weights = 
        new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;

    for (unsigned i = 0; i < 10; i++) {

        char* input = static_cast<char*>(malloc(model->input_size()));
        char* output = static_cast<char*>(malloc(model->output_size()));

        TestInferAction* infer = 
            new TestInferAction(clockwork, 0, util::now(), util::now()+1000000000, input, output);

        infer->submit();
        infer->await();
        infer->check_success(true);

        delete infer;
    }
}

TEST_CASE("Infer Action Multiple Concurrent", "[action] [infer_action]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    TestLoadWeightsAction* load_weights = 
        new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;

    std::vector<TestInferAction*> infers;

    for (unsigned i = 0; i < 10; i++) {

        char* input = static_cast<char*>(malloc(model->input_size()));
        char* output = static_cast<char*>(malloc(model->output_size()));

        TestInferAction* infer = 
            new TestInferAction(clockwork, 0, util::now(), util::now()+1000000000, input, output);

        infers.push_back(infer);
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
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    TestLoadWeightsAction* load_weights = 
        new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;

    char* input = static_cast<char*>(malloc(model->input_size()));
    char* output = static_cast<char*>(malloc(model->output_size()));

    TestInferAction* infer = 
        new TestInferAction(clockwork, 0, util::now(), util::now()+1000000000, input, output);

    infer->submit();
    infer->await();
    infer->check_success(true);

    delete infer;

    TestEvictWeightsAction* evict_weights = 
        new TestEvictWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(true);

    delete evict_weights;

    TestInferAction* infer2 = 
        new TestInferAction(clockwork, 0, util::now(), util::now()+1000000000, input, output);

    infer2->submit();
    infer2->await();
    infer2->check_success(false, actionErrorModelWeightsNotPresent);

    delete infer2;
}