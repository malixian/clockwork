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

class TestLoadWeightsAction : public LoadWeightsAction {
public:
    std::atomic_bool is_success = false;
    std::atomic_bool is_error = false;
    int status_code;
    std::string error_message;

    TestLoadWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest) : 
        LoadWeightsAction(runtime, model_id, earliest, latest) {}

    void success() {
        is_success = true;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
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

TEST_CASE("Load Weights Action", "[action]") {
    Model* model = make_model_for_action();
    ClockworkRuntime* clockwork = make_runtime();
    clockwork->manager->models->put(0, new RuntimeModel(model));

    TestLoadWeightsAction* action = 
        new TestLoadWeightsAction(clockwork, 0, util::now(), util::now()+1000000000);

    action->submit();

    while ((!action->is_success) && (!action->is_error));

    REQUIRE(action->is_success);
}