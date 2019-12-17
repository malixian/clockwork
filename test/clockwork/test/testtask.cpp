#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>
#include <memory>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/task.h"
#include <catch2/catch.hpp>

using namespace clockwork;
using namespace clockwork::model;

class TestLoadModelFromDiskTask : public LoadModelFromDiskTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    RuntimeModel* rm;
    int status_code;
    std::string error_message;

    TestLoadModelFromDiskTask(MemoryManager* cache, int model_id, std::string model_path, uint64_t earliest, uint64_t latest) : 
            LoadModelFromDiskTask(cache, model_id, model_path, earliest, latest) {
    }

    void run(cudaStream_t stream) {
        try {
            LoadModelFromDiskTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void success(RuntimeModel* rm) {
        is_success = true;
        this->rm = rm;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

class TestLoadWeightsTask : public LoadWeightsTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    RuntimeModel* rm;
    int status_code;
    std::string error_message;

    TestLoadWeightsTask(MemoryManager* cache, int model_id, uint64_t earliest, uint64_t latest) : LoadWeightsTask(cache, model_id, earliest, latest) {}

    void run(cudaStream_t stream) {
        try {
            LoadWeightsTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void process_completion() {
        try {
            LoadWeightsTask::process_completion();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }    
    }

    void success(RuntimeModel* rm) {
        is_success = true;
        this->rm = rm;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

class TestEvictWeightsTask : public EvictWeightsTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    RuntimeModel* rm;
    int status_code;
    std::string error_message;

    TestEvictWeightsTask(MemoryManager* cache, int model_id, uint64_t earliest, uint64_t latest) : EvictWeightsTask(cache, model_id, earliest, latest) {}

    void run(cudaStream_t stream) {
        try {
            EvictWeightsTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void success(RuntimeModel* rm) {
        is_success = true;
        this->rm = rm;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};


class TestCopyInputTask : public CopyInputTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    int status_code;
    std::string error_message;
    RuntimeModel* rm;
    char* io_memory;

    TestCopyInputTask(MemoryManager* cache, int model_id, uint64_t earliest, uint64_t latest, unsigned batch_size, size_t input_size, char* input) : CopyInputTask(cache, model_id, earliest, latest, batch_size, input_size, input), io_memory(nullptr) {}

    void run(cudaStream_t stream) {
        try {
            CopyInputTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void process_completion() {
        try {
            CopyInputTask::process_completion();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }    
    }

    void success(RuntimeModel* rm, char* io_memory) {
        is_success = true;
        this->rm = rm;
        this->io_memory = io_memory;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

class TestInferTask : public ExecTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    int status_code;
    std::string error_message;

    TestInferTask(RuntimeModel* rm, MemoryManager* cache, uint64_t earliest, uint64_t latest, unsigned batch_size, char* io_memory) : ExecTask(rm, cache, earliest, latest, batch_size, io_memory) {}

    void run(cudaStream_t stream) {
        try {
            ExecTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void process_completion() {
        try {
            ExecTask::process_completion();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }    
    }

    void success() {
        is_success = true;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

class TestCopyOutputTask : public CopyOutputTask {
public:
    bool is_success = false;
    bool is_error = false;
    bool is_cancelled = false;
    int status_code;
    std::string error_message;
    char* output;

    TestCopyOutputTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, unsigned batch_size, char* io_memory) : CopyOutputTask(rm, manager, earliest, latest, batch_size, io_memory) {}

    void run(cudaStream_t stream) {
        try {
            CopyOutputTask::run(stream);
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }
    }

    void cancel() {
        is_cancelled = true;
    }

    void process_completion() {
        try {
            CopyOutputTask::process_completion();
        } catch (TaskError &error) {
            this->error(error.status_code, error.message);
        }    
    }

    void success(char* output) {
        is_success = true;
        this->output = output;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

// Models get deleted by the MemoryManager
Model* make_model() {
    std::string f = clockwork::util::get_example_model();
    Model* model = Model::loadFromDisk(f+".1.so", f+".1.clockwork", f+".clockwork_params");
    return model;
}

// Models get deleted by the MemoryManager
BatchedModel* make_batched_model(int batch_size, Model* model) {
    std::vector<std::pair<unsigned, Model*>> models = {{batch_size, model}};
    BatchedModel* batched = new BatchedModel(model->weights_size, model->weights_pinned_host_memory, models);
    batched->instantiate_models_on_host();
    batched->instantiate_models_on_device();
    return batched;    
}

class Autostream {
public:
    cudaStream_t stream;
    Autostream() {
        REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);        
    }
    ~Autostream() {
        REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    }
};

std::shared_ptr<MemoryManager> make_manager(
        size_t weights_cache_size, size_t weights_page_size, 
        size_t io_pool_size,
        size_t workspace_pool_size,
        size_t host_io_pool_size) {
    return std::make_shared<MemoryManager>(weights_cache_size, weights_page_size, io_pool_size, workspace_pool_size, host_io_pool_size);
}

std::shared_ptr<MemoryManager> make_manager() {
    return make_manager(
        1024L * 1024L * 1024L,  16L * 1024L * 1024L,
        128L * 1024L * 1024L,
        256L * 1024L * 1024L,
        128L * 1024L * 1024L);
}

TEST_CASE("Load Model From Disk", "[task] [loadmodel]") {
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    int model_id = 0;
    std::string model_path = clockwork::util::get_example_model();

    TestLoadModelFromDiskTask task(manager.get(), model_id, model_path, util::now(), util::now()+1000000000);

    task.run(stream->stream);

    INFO(task.status_code << ": " << task.error_message);
    REQUIRE(!task.is_error);
    REQUIRE(task.is_success);
}

TEST_CASE("Load Non-Existent Model From Disk", "[task] [loadmodel]") {
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    int model_id = 0;
    std::string model_path = clockwork::util::get_example_model() + "bad";

    TestLoadModelFromDiskTask task(manager.get(),model_id, model_path, util::now(), util::now()+1000000000);

    task.run(stream->stream);

    INFO(task.status_code << ": " << task.error_message);
    REQUIRE(task.is_error);
    REQUIRE(!task.is_success);
}

TEST_CASE("Load Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    uint64_t now = util::now();
    TestLoadWeightsTask task(manager.get(),0, now, now+1000000000);

    REQUIRE(task.eligible() == now);


    task.run(stream->stream);

    REQUIRE(!task.is_complete());

    while (!task.is_complete());

    task.process_completion();

    REQUIRE(task.is_success);
    REQUIRE(!task.is_error);
}

TEST_CASE("Load Weights Nonexistent Model", "[task]") {
    Model* model = make_model();
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    uint64_t now = util::now();
    TestLoadWeightsTask task(manager.get(),0, now, now+1000000000);

    REQUIRE(task.eligible() == now);


    task.run(stream->stream);
    REQUIRE(task.is_error);
    REQUIRE(!task.is_success);
    REQUIRE(task.status_code == actionErrorUnknownModel);
}

TEST_CASE("Load Weights Earliest", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask task(manager.get(),0, now+1000000000, now+1000000000);

    task.run(stream->stream);

    REQUIRE(task.is_complete());
    REQUIRE(!task.is_success);
    REQUIRE(task.is_error);
    REQUIRE(task.status_code == actionErrorRuntimeError);
}

TEST_CASE("Load Weights Latest", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask task(manager.get(),0, 0, now - 1000000);

    task.run(stream->stream);

    REQUIRE(task.is_complete());
    REQUIRE(!task.is_success);
    REQUIRE(task.is_error);
    REQUIRE(task.status_code == actionErrorCouldNotStartInTime);
}

TEST_CASE("Load Weights Insufficient Cache", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager(16 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024, 64 * 1024 * 1024,64 * 1024 * 1024);
    manager->models->put(0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask task(manager.get(),0, 0, now + 1000000000L);

    task.run(stream->stream);

    REQUIRE(task.is_complete());
    REQUIRE(!task.is_success);
    REQUIRE(task.is_error);
    REQUIRE(task.status_code == actionErrorRuntimeError);
}

TEST_CASE("Load Weights Version Update", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask task(manager.get(),0, 0, now + 1000000000L);

    task.run(stream->stream);
    rm->lock();

    REQUIRE(!task.is_complete());

    rm->version++;
    rm->unlock();

    while (!task.is_complete());
    task.process_completion();

    REQUIRE(!task.is_success);
    REQUIRE(task.is_error);
    REQUIRE(task.status_code == actionErrorWeightsInUse);
}

TEST_CASE("Double Load Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    // Load weights 1
    TestLoadWeightsTask load1(manager.get(),0, 0, util::now() + 1000000000L);
    load1.run(stream->stream);

    rm->lock();

    int invalid_version = rm->version;
    std::shared_ptr<Allocation> invalid_weights = rm->weights;

    rm->unlock();

    // Load weights 2
    TestLoadWeightsTask load2(manager.get(),0, 0, util::now() + 1000000000L);
    load2.run(stream->stream);

    while (!load1.is_complete());
    load1.process_completion();
    while (!load2.is_complete());
    load2.process_completion();
    
    REQUIRE(!load1.is_success);
    REQUIRE(load1.is_error);
    REQUIRE(load2.is_success);
    REQUIRE(!load2.is_error);
}



TEST_CASE("Evict Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    // Load weights
    uint64_t now = util::now();
    TestLoadWeightsTask load(manager.get(),0, 0, util::now() + 1000000000L);
    load.run(stream->stream);
    while (!load.is_complete());
    load.process_completion();
    
    REQUIRE(load.is_success);
    REQUIRE(!load.is_error);

    // Now evict them
    TestEvictWeightsTask evict(manager.get(),0, 0, util::now() + 1000000000);
    evict.run(stream->stream);

    REQUIRE(evict.is_success);
    REQUIRE(!evict.is_error);
}

TEST_CASE("Evict Non-Existent Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    // Evict weights
    TestEvictWeightsTask evict(manager.get(),0, 0, util::now() + 1000000000);
    evict.run(stream->stream);

    REQUIRE(!evict.is_success);
    REQUIRE(evict.is_error);
    REQUIRE(evict.status_code == actionErrorModelWeightsNotPresent);
}

TEST_CASE("Double Evict", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    // Load weights
    uint64_t now = util::now();
    TestLoadWeightsTask load(manager.get(),0, 0, util::now() + 1000000000L);
    load.run(stream->stream);
    while (!load.is_complete());
    load.process_completion();
    
    REQUIRE(load.is_success);
    REQUIRE(!load.is_error);

    // Now evict them
    TestEvictWeightsTask evict(manager.get(),0, 0, util::now() + 1000000000);
    evict.run(stream->stream);

    REQUIRE(evict.is_success);
    REQUIRE(!evict.is_error);

    // Now evict them
    TestEvictWeightsTask evict2(manager.get(),0, 0, util::now() + 1000000000);
    evict2.run(stream->stream);

    REQUIRE(!evict2.is_success);
    REQUIRE(evict2.is_error);
    REQUIRE(evict2.status_code == actionErrorModelWeightsNotPresent);
}

TEST_CASE("Evict Weights Nonexistent Model", "[task]") {
    Model* model = make_model();
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    uint64_t now = util::now();
    TestEvictWeightsTask task(manager.get(),0, now, now+1000000000);

    REQUIRE(task.eligible() == now);

    task.run(stream->stream);
    REQUIRE(task.is_error);
    REQUIRE(!task.is_success);
    REQUIRE(task.status_code == actionErrorUnknownModel);
}

TEST_CASE("Copy Input", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask copyinput(manager.get(),0, 0, util::now() + 1000000000, 1, model->input_size(), input);
    copyinput.run(stream->stream);
    while (!copyinput.is_complete());
    copyinput.process_completion();

    INFO("Error " << copyinput.status_code << ": " << copyinput.error_message);
    REQUIRE(copyinput.is_success);
    REQUIRE(!copyinput.is_error);

    free(input);
}

TEST_CASE("Copy Input Wrong Size", "[task] [wrongsize]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    char* input = static_cast<char*>(malloc(10));

    TestCopyInputTask copyinput(manager.get(),0, 0, util::now() + 1000000000, 1, 10, input);
    copyinput.run(stream->stream);
    while (!copyinput.is_complete());

    REQUIRE(!copyinput.is_success);
    REQUIRE(copyinput.is_error);
    REQUIRE(copyinput.status_code == actionErrorInvalidInput);

    free(input);
}

TEST_CASE("Copy Input Nonexistent Model", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    char* input = static_cast<char*>(malloc(model->input_size()));

    uint64_t now = util::now();
    TestCopyInputTask copyinput(manager.get(),0, now, util::now() + 1000000000, 1, model->input_size(), input);

    REQUIRE(copyinput.eligible() == now);


    copyinput.run(stream->stream);
    REQUIRE(copyinput.is_error);
    REQUIRE(!copyinput.is_success);
    REQUIRE(copyinput.status_code == actionErrorUnknownModel);
}


TEST_CASE("Input and Infer", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);


    TestLoadWeightsTask loadweights(manager.get(),0, 0, util::now()+1000000000);
    loadweights.run(stream->stream);
    REQUIRE(!loadweights.is_error);
    
    while (!loadweights.is_complete());
    loadweights.process_completion();

    REQUIRE(loadweights.is_success);
    REQUIRE(!loadweights.is_error);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask copyinput(manager.get(),0, 0, util::now() + 1000000000, 1, model->input_size(), input);
    copyinput.run(stream->stream);
    REQUIRE(!copyinput.is_error);
    
    while (!copyinput.is_complete());
    copyinput.process_completion();

    REQUIRE(copyinput.is_success);
    REQUIRE(!copyinput.is_error);

    TestInferTask infer(rm, manager.get(),0, util::now() + 1000000000, 1, copyinput.io_memory);
    infer.run(stream->stream);
    REQUIRE(!infer.is_error);

    while (!infer.is_complete());
    infer.process_completion();

    REQUIRE(infer.is_success);
    REQUIRE(!infer.is_error);

    free(input);
}

TEST_CASE("Infer Without Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(make_batched_model(1, model));
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();
    manager->models->put(0, rm);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask copyinput(manager.get(),0, 0, util::now() + 1000000000, 1, model->input_size(), input);
    copyinput.run(stream->stream);
    REQUIRE(!copyinput.is_error);
    
    while (!copyinput.is_complete());
    copyinput.process_completion();

    REQUIRE(copyinput.is_success);
    REQUIRE(!copyinput.is_error);

    TestInferTask infer(rm, manager.get(),0, util::now() + 1000000000, 1, copyinput.io_memory);
    infer.run(stream->stream);
    REQUIRE(!infer.is_success);
    REQUIRE(infer.is_error);
    REQUIRE(infer.status_code == actionErrorModelWeightsNotPresent);

    free(input);
}



TEST_CASE("Input Infer and Output", "[task]") {
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    std::string model_path = clockwork::util::get_example_model();

    TestLoadModelFromDiskTask loadmodel(manager.get(),0, model_path, util::now(), util::now()+1000000000);

    loadmodel.run(stream->stream);
    REQUIRE(loadmodel.is_success);
    REQUIRE(!loadmodel.is_error);

    RuntimeModel* rm = manager->models->get(0);
    REQUIRE(rm != nullptr);
    model::BatchedModel* model = rm->model;

    TestLoadWeightsTask loadweights(manager.get(),0, 0, util::now()+1000000000);
    loadweights.run(stream->stream);
    REQUIRE(!loadweights.is_error);
    
    while (!loadweights.is_complete());
    loadweights.process_completion();

    REQUIRE(loadweights.is_success);
    REQUIRE(!loadweights.is_error);

    char* input = static_cast<char*>(malloc(model->input_size(1)));

    TestCopyInputTask copyinput(manager.get(),0, 0, util::now() + 1000000000, 1, model->input_size(1), input);
    copyinput.run(stream->stream);
    REQUIRE(!copyinput.is_error);
    
    while (!copyinput.is_complete());
    copyinput.process_completion();

    REQUIRE(copyinput.is_success);
    REQUIRE(!copyinput.is_error);
    REQUIRE(copyinput.io_memory != nullptr);

    TestInferTask infer(rm, manager.get(),0, util::now() + 1000000000, 1, copyinput.io_memory);
    infer.run(stream->stream);
    REQUIRE(!infer.is_error);

    while (!infer.is_complete());
    infer.process_completion();

    REQUIRE(infer.is_success);
    REQUIRE(!infer.is_error);

    TestCopyOutputTask copyoutput(rm, manager.get(),0, util::now() + 1000000000, 1, copyinput.io_memory);
    copyoutput.run(stream->stream);
    REQUIRE(!copyoutput.is_error);

    while (!copyoutput.is_complete());
    copyoutput.process_completion();

    REQUIRE(copyoutput.is_success);
    REQUIRE(!copyoutput.is_error);

    free(input);
}

TEST_CASE("Input Infer and Output Batched", "[task]") {
    auto stream = std::make_shared<Autostream>();
    auto manager = make_manager();

    std::string model_path = clockwork::util::get_example_model("resnet18_tesla-m40");

    TestLoadModelFromDiskTask loadmodel(manager.get(),0, model_path, util::now(), util::now()+1000000000);

    loadmodel.run(stream->stream);
    REQUIRE(loadmodel.is_success);
    REQUIRE(!loadmodel.is_error);

    RuntimeModel* rm = manager->models->get(0);
    REQUIRE(rm != nullptr);
    model::BatchedModel* model = rm->model;

    TestLoadWeightsTask loadweights(manager.get(),0, 0, util::now()+1000000000);
    loadweights.run(stream->stream);
    REQUIRE(!loadweights.is_error);
    
    while (!loadweights.is_complete());
    loadweights.process_completion();

    REQUIRE(loadweights.is_success);
    REQUIRE(!loadweights.is_error);

    for (unsigned batch_size = 1; batch_size <= 16; batch_size++) {

        char* input = manager->host_io_pool->alloc(model->input_size(batch_size));

        TestCopyInputTask copyinput(manager.get(),0, 0, util::now() + 1000000000, batch_size, model->input_size(batch_size), input);
        copyinput.run(stream->stream);
        INFO("Error " << copyinput.status_code << ": " << copyinput.error_message);
        REQUIRE(!copyinput.is_error);
        
        while (!copyinput.is_complete());
        copyinput.process_completion();

        INFO("Error " << copyinput.status_code << ": " << copyinput.error_message);
        REQUIRE(!copyinput.is_error);
        REQUIRE(copyinput.is_success);
        REQUIRE(copyinput.io_memory != nullptr);

        TestInferTask infer(rm, manager.get(),0, util::now() + 1000000000, batch_size, copyinput.io_memory);
        infer.run(stream->stream);
        INFO("Error " << infer.status_code << ": " << infer.error_message);
        REQUIRE(!infer.is_error);

        while (!infer.is_complete());
        infer.process_completion();

        INFO("Error " << infer.status_code << ": " << infer.error_message);
        REQUIRE(!infer.is_error);
        REQUIRE(infer.is_success);

        TestCopyOutputTask copyoutput(rm, manager.get(),0, util::now() + 1000000000, batch_size, copyinput.io_memory);
        copyoutput.run(stream->stream);
        INFO("Error " << copyoutput.status_code << ": " << copyoutput.error_message);
        REQUIRE(!copyoutput.is_error);

        while (!copyoutput.is_complete());
        copyoutput.process_completion();

        INFO("Error " << copyoutput.status_code << ": " << copyoutput.error_message);
        REQUIRE(!copyoutput.is_error);
        REQUIRE(copyoutput.is_success);

        manager->host_io_pool->free(copyoutput.output);
        manager->workspace_pool->free(copyinput.io_memory);
        manager->host_io_pool->free(input);
    }
}
