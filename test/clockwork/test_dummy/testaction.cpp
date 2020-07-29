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
    weights_load_time_nanos = modeldata[0].weights_measurement;
    for (ModelDataDummy &d : modeldata) {
        supported_batch_sizes.push_back(d.batch_size);
        batch_size_exec_times_nanos.push_back(d.exec_measurement);
    }

    //deserialize the model metadata
    model::PageMappedModelDef* spec = new model::PageMappedModelDef();
    model::PageMappedModelDef::ReadFrom(modeldata[0].serialized_spec, *spec);

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

    TestLoadModelFromDiskDummy load_model(clockwork->manager, load_model_from_disk_action());

    load_model.run();
    load_model.await();
    load_model.check_success(true);
}