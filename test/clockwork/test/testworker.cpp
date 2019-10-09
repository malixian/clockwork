#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/worker.h"
#include <catch2/catch.hpp>
#include "clockwork/test/actions.h"
#include "tbb/concurrent_queue.h"

using namespace clockwork;
using namespace clockwork::model;

class TestController : public workerapi::Controller {
public:
    tbb::concurrent_queue<std::shared_ptr<workerapi::Result>> results;

    void sendResult(std::shared_ptr<workerapi::Result> result) {
        results.push(result);
    }

    std::shared_ptr<workerapi::Result> awaitResult() {
        std::shared_ptr<workerapi::Result> result;
        while (!results.try_pop(result));
        return result;
    }

    void expect(int expected_status_code) {
        std::shared_ptr<workerapi::Result> result = awaitResult();
        INFO("id=" << result->id << " type=" << result->action_type << " status=" << result->status);
        if (result->status != actionSuccess) {
            auto error = std::static_pointer_cast<workerapi::ErrorResult>(result);
            INFO(error->message);
            REQUIRE(result->status == expected_status_code);
        } else {
            REQUIRE(result->status == expected_status_code);            
        }
    }
};

TEST_CASE("Test Worker", "[worker]") {
    ClockworkWorker* worker = new ClockworkWorker();
    TestController* controller = new TestController();
    worker->controller = controller;

    auto load_model = load_model_from_disk_action();

    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};

    worker->sendActions(actions);
    controller->expect(actionSuccess);

    worker->shutdown(true);
    delete worker;
}

TEST_CASE("Test Infer No Weights", "[worker] [noweights]") {
    ClockworkWorker* worker = new ClockworkWorker();
    TestController* controller = new TestController();
    worker->controller = controller;

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker->sendActions(actions);
    controller->expect(actionSuccess);

    auto infer = infer_action2(worker);
    actions = {infer};
    worker->sendActions(actions);
    controller->expect(actionErrorModelWeightsNotPresent);

    worker->shutdown(true);
    delete worker;
}

TEST_CASE("Test Infer Weights Not There Yet", "[worker] [noweights]") {
    ClockworkWorker* worker = new ClockworkWorker();
    TestController* controller = new TestController();
    worker->controller = controller;

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker->sendActions(actions);
    controller->expect(actionSuccess);

    auto load_weights = load_weights_action();
    auto infer = infer_action2(worker);
    actions = {load_weights, infer};
    worker->sendActions(actions);
    controller->expect(actionErrorModelWeightsNotPresent);
    controller->expect(actionSuccess);

    worker->shutdown(true);
    delete worker;
}

TEST_CASE("Test Infer Invalid Input", "[worker] [invalid]") {
    ClockworkWorker* worker = new ClockworkWorker();
    TestController* controller = new TestController();
    worker->controller = controller;

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker->sendActions(actions);
    controller->expect(actionSuccess);

    auto infer = infer_action2(worker);
    infer->input_size = 10;
    infer->input = nullptr;
    actions = {infer};
    worker->sendActions(actions);
    controller->expect(actionErrorInvalidInput);

    worker->shutdown(true);
    delete worker;
}

TEST_CASE("Test Infer Invalid Input Size", "[worker] [invalid]") {
    ClockworkWorker* worker = new ClockworkWorker();
    TestController* controller = new TestController();
    worker->controller = controller;

    auto load_model = load_model_from_disk_action();
    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};
    worker->sendActions(actions);
    controller->expect(actionSuccess);

    auto infer = infer_action2(worker);
    infer->input_size = 100;
    infer->input = static_cast<char*>(malloc(100));
    actions = {infer};
    worker->sendActions(actions);
    controller->expect(actionErrorInvalidInput);

    worker->shutdown(true);
    delete worker;
}

TEST_CASE("Test Worker E2E Simple", "[worker] [e2esimple]") {
    ClockworkWorker* worker = new ClockworkWorker();
    TestController* controller = new TestController();
    worker->controller = controller;

    std::vector<std::shared_ptr<workerapi::Action>> actions;

    auto load_model = load_model_from_disk_action();
    actions = {load_model};
    worker->sendActions(actions);
    controller->expect(actionSuccess);

    auto load_weights = load_weights_action();
    actions = {load_weights};
    worker->sendActions(actions);
    controller->expect(actionSuccess);

    auto infer = infer_action2(worker);
    actions = {infer};
    worker->sendActions(actions);
    controller->expect(actionSuccess);

    auto evict_weights = evict_weights_action();
    actions = {evict_weights};
    worker->sendActions(actions);
    controller->expect(actionSuccess);

    worker->shutdown(true);
    delete worker;
}

TEST_CASE("Test Worker E2E Timed Success", "[worker]") {
    ClockworkWorker* worker = new ClockworkWorker();
    TestController* controller = new TestController();
    worker->controller = controller;

    auto load_model = load_model_from_disk_action();

    auto load_weights = load_weights_action();
    load_weights->earliest = load_model->earliest + 1000000000UL;
    load_weights->latest = load_weights->earliest + 100000000UL;

    auto infer = infer_action2(worker);
    infer->earliest = load_weights->earliest + 20000000;
    infer->latest = infer->earliest + 100000000UL;

    auto evict_weights = evict_weights_action();
    evict_weights->earliest = infer->earliest + 10000000;
    evict_weights->latest = evict_weights->earliest + 100000000UL;

    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model, load_weights, infer, evict_weights};
    worker->sendActions(actions);

    for (unsigned i = 0; i < 4; i++) {
        controller->expect(actionSuccess);
    }

    worker->shutdown(true);
    delete worker;
}