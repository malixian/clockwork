#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/modeldef.h"
#include "clockwork/test_dummy/actions.h"
#include "clockwork/dummy/clockwork/worker_dummy.h"

using namespace clockwork;

std::shared_ptr<ClockworkRuntimeDummy> make_runtime() {
    return std::make_shared<ClockworkRuntimeWrapperDummy>();
}

RuntimeModelDummy* make_model_for_action(){
    std::string f = clockwork::util::get_example_model();

    // Load data for batch sizes and extract performance profile
    std::vector<unsigned> supported_batch_sizes;
    std::vector<uint64_t> batch_size_exec_times_nanos;
    uint64_t weights_load_time_nanos;

    std::vector<ModelDataDummy> modeldata = loadModelDataDummy(f);
    weights_load_time_nanos = modeldata[GPU_ID_0].weights_measurement;
    for (ModelDataDummy &d : modeldata) {
        supported_batch_sizes.push_back(d.batch_size);
        batch_size_exec_times_nanos.push_back(d.exec_measurement);
    }

    //deserialize the model metadata
    model::PageMappedModelDef* spec = new model::PageMappedModelDef();
    model::PageMappedModelDef::ReadFrom(modeldata[GPU_ID_0].serialized_spec, *spec);

    //Extract model metadata
    unsigned weights_pages_count = spec->weights_pages.size();
    uint64_t weights_size = weights_pages_count*spec->configured_weights_page_size;
    size_t inputs_size = 0;
    size_t outputs_size = 0;
    for (auto &input : spec->inputs) {
        inputs_size += input.size;
    }
    for (auto &output : spec->outputs) {
        outputs_size += output.size;
    }

    workerapi::ModelInfo* modelInfo = new workerapi::ModelInfo();
    modelInfo->id = 0;
    modelInfo->source = f;
    modelInfo->input_size = inputs_size;
    modelInfo->output_size = outputs_size;
    modelInfo->supported_batch_sizes = supported_batch_sizes;
    modelInfo->weights_size = weights_size;
    modelInfo->num_weights_pages = spec->configured_weights_page_size;
    modelInfo->weights_load_time_nanos = weights_load_time_nanos;
    modelInfo->batch_size_exec_times_nanos = batch_size_exec_times_nanos;
    return new RuntimeModelDummy(modelInfo,GPU_ID_0,weights_pages_count);
}

TEST_CASE("Load Model From Disk Action", "[action] [loadmodel_action]") {
    auto clockwork = make_runtime();

    TestLoadModelFromDiskDummy load_model(clockwork.get(), load_model_from_disk_action());

    load_model.submit();
    load_model.await();
    load_model.check_success(true);
}

TEST_CASE("Load Model From Disk Action Multiple", "[action] [loadmodel_action]") {
    auto clockwork = make_runtime();

    for (unsigned i = 0; i < 10; i++) {
        auto action = load_model_from_disk_action();
        action->model_id = i;

        TestLoadModelFromDiskDummy load_model(clockwork.get(), action);

        load_model.submit();
        load_model.await();
        load_model.check_success(true);
    }
}

TEST_CASE("Load Non-Existent Model From Disk", "[action] [loadmodel_action]") {
    auto clockwork = make_runtime();
    auto action = load_model_from_disk_action();
    action->model_path = "";
    TestLoadModelFromDiskDummy load_model(clockwork.get(),action);

    load_model.submit();
    load_model.await();
    load_model.check_success(false, actionErrorInvalidModelPath);
}


TEST_CASE("Load Weights Action", "[action] [loadweights_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);    
}

TEST_CASE("Load Weights Action Multiple", "[action] [loadweights_multiple]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    for (unsigned i = 0; i < 5; i++) {
        TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

        load_weights.submit();
        load_weights.await();
        load_weights.check_success(true);
    }
}

TEST_CASE("Load Weights Action Invalid Model", "[action] [loadweights_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(false, loadWeightsUnknownModel);
}
/*
TEST_CASE("Load Weights Earliest", "[action] [loadweights_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    uint64_t now = util::now();
    auto action = load_weights_action();
    action->earliest = now+1000000000;
    action->latest = now+1000000000;

    TestLoadWeightsDummy load_weights(clockwork.get(), action);

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(false, actionErrorRuntimeError);
}*/

TEST_CASE("Load Weights Latest", "[action] [loadweights_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    uint64_t now = util::now();
    auto action = load_weights_action();
    action->earliest = 0;
    action->latest = now - 1000000;

    TestLoadWeightsDummy load_weights(clockwork.get(), action);

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(false, actionErrorCouldNotStartInTime);
}

TEST_CASE("Load Weights Insufficient Cache", "[action] [loadweights_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);
    clockwork->manager->weights_caches[GPU_ID_0]->n_free_pages = 0;

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(false, loadWeightsInsufficientCache);
}

TEST_CASE("Load Weights Version Update", "[action] [loadweights_action]") {
    RuntimeModelDummy* rm = make_model_for_action();
    rm->modelinfo->weights_load_time_nanos = 100000;//Hard code let loadweight finish later
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, rm);

    auto action = load_weights_action();
    TestLoadWeightsDummy load_weights(clockwork.get(), action);
    load_weights.run();

    rm->lock();
    bool flag = (!load_weights.is_success) && (!load_weights.is_error);
    REQUIRE(flag);
    rm->version++;
    rm->unlock();

    load_weights.await();
    load_weights.check_success(false, loadWeightsConcurrentModification);
}

//here

TEST_CASE("Load Evict Weights Action", "[action] [evict_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    TestEvictWeightsDummy evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(true);
}

TEST_CASE("Evict without Weights Action", "[action] [evict_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestEvictWeightsDummy evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(false, evictWeightsNotInCache);
}

TEST_CASE("Infer Action", "[action] [infer_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    TestInferDummy infer(clockwork.get(), infer_action(1, model));

    infer.submit();
    infer.await();
    infer.check_success(true);
}

TEST_CASE("Infer Action Wrong Input Size", "[action] [nomodel]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    auto action = infer_action();
    action->input_size = 10;
    TestInferDummy infer(clockwork.get(), action);

    infer.submit();
    infer.await();
    infer.check_success(false, copyInputInvalidInput);
}

TEST_CASE("Infer Action Nonexistent Model", "[action] [nomodel]") {
    auto clockwork = make_runtime();

    TestInferDummy infer(clockwork.get(), infer_action());

    infer.submit();
    infer.await();
    infer.check_success(false, copyInputUnknownModel);
}

TEST_CASE("Infer Action Nonexistent Weights", "[action] [noweights]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestInferDummy infer(clockwork.get(), infer_action());

    infer.submit();
    infer.await();
    infer.check_success(false, execWeightsMissing);
}

TEST_CASE("Infer Action Multiple", "[action] [infer_action_multiple]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    for (unsigned i = 0; i < 2; i++) {
        TestInferDummy infer(clockwork.get(), infer_action(1, model));

        infer.submit();
        infer.await();
        infer.check_success(true);
    }
}

TEST_CASE("Make Many Models", "[action] [models]") {
    std::vector<RuntimeModelDummy*> models;
    for (unsigned i = 0; i < 30; i++) {
        models.push_back(make_model_for_action());
    }
    for (RuntimeModelDummy* model : models) {
        delete model;
    }
}//Q

TEST_CASE("Infer Action Multiple Concurrent", "[action] [infer_action_concurrent]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    std::vector<std::shared_ptr<TestInferDummy>> infers;

    for (unsigned i = 0; i < 2; i++) {
        infers.push_back(std::make_shared<TestInferDummy>(clockwork.get(), infer_action(1, model)));
    }

    for (auto infer : infers) {
        infer->submit();
    }

    for (auto infer : infers) {
        infer->await();
        infer->check_success(true);
    }    
}

TEST_CASE("Infer after Evict Action", "[action] [evict_action]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    TestLoadWeightsDummy load_weights(clockwork.get(), load_weights_action());

    load_weights.submit();
    load_weights.await();
    load_weights.check_success(true);

    TestInferDummy infer(clockwork.get(), infer_action(1, model));

    infer.submit();
    infer.await();
    infer.check_success(true);

    TestEvictWeightsDummy evict_weights(clockwork.get(), evict_weights_action());

    evict_weights.submit();
    evict_weights.await();
    evict_weights.check_success(true);

    TestInferDummy infer2(clockwork.get(), infer_action(1, model));

    infer2.submit();
    infer2.await();
    infer2.check_success(false, execWeightsMissing);   
}

TEST_CASE("Actions E2E", "[action] [e2e]") {
    auto clockwork = make_runtime();

    auto load_model = new TestLoadModelFromDiskDummy(clockwork.get(), load_model_from_disk_action());

    load_model->submit();
    load_model->await();
    load_model->check_success(true);

    delete load_model;

    auto load_weights = new TestLoadWeightsDummy(clockwork.get(), load_weights_action());

    load_weights->submit();
    load_weights->await();
    load_weights->check_success(true);

    delete load_weights;
    
    RuntimeModelDummy* model = clockwork->manager->models->get(0, GPU_ID_0);
    auto infer = new TestInferDummy(clockwork.get(), infer_action(1, model));

    infer->submit();
    infer->await();
    infer->check_success(true);

    delete infer;

    auto evict_weights = new TestEvictWeightsDummy(clockwork.get(), evict_weights_action());

    evict_weights->submit();
    evict_weights->await();
    evict_weights->check_success(true);

    delete evict_weights;

    auto infer2 = new TestInferDummy(clockwork.get(), infer_action(1, model));

    infer2->submit();
    infer2->await();
    infer2->check_success(false, execWeightsMissing);

    delete infer2;

    
}

TEST_CASE("Task Cancelled After Shutdown", "[action] [shutdown]") {
    RuntimeModelDummy* model = make_model_for_action();
    auto clockwork = make_runtime();
    clockwork->manager->models->put(0, GPU_ID_0, model);

    auto action = load_weights_action();
    //action->earliest = util::now() + 1000000000UL; // + 10seconds
    //action->latest = action->earliest; //Q

    auto load_weights = new TestLoadWeightsDummy(clockwork.get(), action);   

    load_weights->submit();

    clockwork = nullptr; // destroys the runtime

    load_weights->await();
    load_weights->check_success(false, actionCancelled);

    delete load_weights;
}

class TestLoadModelFromDiskActionThatDeletesItself : public LoadModelFromDiskDummyAction {
public:
    TestAction &action_status;

    ClockworkRuntimeDummy* myRuntime;

    TestLoadModelFromDiskActionThatDeletesItself(ClockworkRuntimeDummy* runtime, std::shared_ptr<workerapi::LoadModelFromDisk> action, 
            TestAction &action_status) : 
        LoadModelFromDiskDummyAction(runtime->manager, action), action_status(action_status) {myRuntime = runtime;}

    void submit(){
    myRuntime->load_model_executor->actions_to_start.push(element{loadmodel->earliest, [this]() {this->run();} ,[this]() {this->error(actionCancelled, "Action cancelled");} });
    }

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
        action_status.setsuccess();
        delete this;
    }

    void error(int status_code, std::string message) {
        auto result = std::make_shared<workerapi::ErrorResult>();
        result->action_type = workerapi::loadModelFromDiskAction;
        result->status = status_code; 
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

    
}//Q

