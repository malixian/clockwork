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

using namespace clockwork;
using namespace clockwork::model;

class TestController : public workerapi::Controller {
public:

    void sendResult(std::shared_ptr<workerapi::Result> result) {
        std::cout << "Received a result " << result->id << " type=" << result->action_type << " status=" << result->status << std::endl;
    }
};

TEST_CASE("Test Worker", "[worker]") {
    ClockworkWorker* worker = new ClockworkWorker();
    worker->controller = new TestController();

    auto load_model = load_model_from_disk_action();

    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model};

    worker->sendActions(actions);

    usleep(1000000);

    delete worker;
}

TEST_CASE("Test Worker E2E", "[worker]") {
    ClockworkWorker* worker = new ClockworkWorker();
    worker->controller = new TestController();

    auto load_model = load_model_from_disk_action();

    auto load_weights = load_weights_action();
    load_weights->earliest = load_model->earliest + 300000000;

    auto infer = infer_action();
    infer->earliest = load_weights->earliest + 20000000;

    auto evict_weights = evict_weights_action();
    evict_weights->earliest = infer->earliest + 10000000;

    std::vector<std::shared_ptr<workerapi::Action>> actions{load_model, load_weights, infer, evict_weights};
    worker->sendActions(actions);

    usleep(1000000);
}