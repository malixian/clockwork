#include <unistd.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include "clockwork/api/worker_api.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/task.h"
#include <catch2/catch.hpp>

using namespace clockwork;
using namespace clockwork::model;

class TestLoadWeightsTask : public LoadWeightsTask {
public:
    bool is_success = false;
    bool is_error = false;
    int status_code;
    std::string error_message;

    TestLoadWeightsTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest) : LoadWeightsTask(rm, cache, earliest, latest) {}

    void success() {
        is_success = true;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

Model* make_model() {
    std::string f = clockwork::util::get_example_model();

    Model* model = Model::loadFromDisk(f+".so", f+".clockwork", f+".clockwork_params");
    model->instantiate_model_on_host();
    model->instantiate_model_on_device();
    return model;
}

PageCache* make_cache(size_t page_size, size_t total_size) {
    void* baseptr;
    REQUIRE(cudaMalloc(&baseptr, total_size) == cudaSuccess);
    bool allow_evictions = false;

    PageCache* cache = new PageCache(static_cast<char*>(baseptr), total_size, page_size, allow_evictions);
    return cache;
}


PageCache* make_cache() {
    size_t page_size = 16 * 1024 * 1024;
    size_t total_size = page_size * 50;
    return make_cache(page_size, total_size);
}

cudaStream_t make_stream() {
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    return stream;
}

TEST_CASE("Load Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    uint64_t now = util::now();
    TestLoadWeightsTask* task = new TestLoadWeightsTask(rm, cache, now, now+1000000000);

    REQUIRE(task->eligible() == now);


    task->run(stream);

    REQUIRE(!task->is_complete());

    while (!task->is_complete());

    task->process_completion();

    REQUIRE(task->is_success);
    REQUIRE(!task->is_error);
}

TEST_CASE("Load Weights Earliest", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    uint64_t now = util::now();

    TestLoadWeightsTask* task = new TestLoadWeightsTask(rm, cache, now+1000000000, now+1000000000);

    task->run(stream);

    REQUIRE(task->is_complete());
    REQUIRE(!task->is_success);
    REQUIRE(task->is_error);
    REQUIRE(task->status_code == actionErrorRuntimeError);

    delete task;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}

TEST_CASE("Load Weights Latest", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    uint64_t now = util::now();

    TestLoadWeightsTask* task = new TestLoadWeightsTask(rm, cache, 0, now - 1000000);

    task->run(stream);

    REQUIRE(task->is_complete());
    REQUIRE(!task->is_success);
    REQUIRE(task->is_error);
    REQUIRE(task->status_code == actionErrorCouldNotStartInTime);

    delete task;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}

TEST_CASE("Load Weights Insufficient Cache", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache(16 * 1024 * 1024, 16 * 1024 * 1024);
    cudaStream_t stream = make_stream();

    uint64_t now = util::now();

    TestLoadWeightsTask* task = new TestLoadWeightsTask(rm, cache, 0, now + 1000000000L);

    task->run(stream);

    REQUIRE(task->is_complete());
    REQUIRE(!task->is_success);
    REQUIRE(task->is_error);
    REQUIRE(task->status_code == actionErrorRuntimeError);

    delete task;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}

TEST_CASE("Load Weights Version Update", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    uint64_t now = util::now();

    TestLoadWeightsTask* task = new TestLoadWeightsTask(rm, cache, 0, now + 1000000000L);

    task->run(stream);
    rm->lock();

    REQUIRE(!task->is_complete());

    rm->version++;
    rm->unlock();

    while (!task->is_complete());
    task->process_completion();

    REQUIRE(!task->is_success);
    REQUIRE(task->is_error);
    REQUIRE(task->status_code == actionErrorWeightsInUse);

    delete task;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}

TEST_CASE("Double Load Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    // Load weights 1
    TestLoadWeightsTask* load1 = new TestLoadWeightsTask(rm, cache, 0, util::now() + 1000000000L);
    load1->run(stream);

    rm->lock();

    int invalid_version = rm->version;
    std::shared_ptr<Allocation> invalid_weights = rm->weights;

    rm->unlock();

    // Load weights 2
    TestLoadWeightsTask* load2 = new TestLoadWeightsTask(rm, cache, 0, util::now() + 1000000000L);
    load2->run(stream);

    while (!load1->is_complete());
    load1->process_completion();
    while (!load2->is_complete());
    load2->process_completion();
    
    REQUIRE(!load1->is_success);
    REQUIRE(load1->is_error);
    REQUIRE(load2->is_success);
    REQUIRE(!load2->is_error);

    delete load1;
    delete load2;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}


class TestEvictWeightsTask : public EvictWeightsTask {
public:
    bool is_success = false;
    bool is_error = false;
    int status_code;
    std::string error_message;

    TestEvictWeightsTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest) : EvictWeightsTask(rm, cache, earliest, latest) {}

    void success() {
        is_success = true;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};


TEST_CASE("Evict Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    // Load weights
    uint64_t now = util::now();
    TestLoadWeightsTask* load = new TestLoadWeightsTask(rm, cache, 0, util::now() + 1000000000L);
    load->run(stream);
    while (!load->is_complete());
    load->process_completion();
    
    REQUIRE(load->is_success);
    REQUIRE(!load->is_error);

    // Now evict them
    TestEvictWeightsTask* evict = new TestEvictWeightsTask(rm, cache, 0, util::now() + 1000000000);
    evict->run(stream);

    REQUIRE(evict->is_success);
    REQUIRE(!evict->is_error);

    delete load;
    delete evict;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}

TEST_CASE("Evict Non-Existent Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    // Evict weights
    TestEvictWeightsTask* evict = new TestEvictWeightsTask(rm, cache, 0, util::now() + 1000000000);
    evict->run(stream);

    REQUIRE(!evict->is_success);
    REQUIRE(evict->is_error);
    REQUIRE(evict->is_error);
    REQUIRE(evict->status_code == actionErrorModelWeightsNotPresent);

    delete evict;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}

TEST_CASE("Double Evict", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    // Load weights
    uint64_t now = util::now();
    TestLoadWeightsTask* load = new TestLoadWeightsTask(rm, cache, 0, util::now() + 1000000000L);
    load->run(stream);
    while (!load->is_complete());
    load->process_completion();
    
    REQUIRE(load->is_success);
    REQUIRE(!load->is_error);

    // Now evict them
    TestEvictWeightsTask* evict = new TestEvictWeightsTask(rm, cache, 0, util::now() + 1000000000);
    evict->run(stream);

    REQUIRE(evict->is_success);
    REQUIRE(!evict->is_error);

    // Now evict them
    TestEvictWeightsTask* evict2 = new TestEvictWeightsTask(rm, cache, 0, util::now() + 1000000000);
    evict2->run(stream);

    REQUIRE(!evict2->is_success);
    REQUIRE(evict2->is_error);
    REQUIRE(evict2->status_code == actionErrorModelWeightsNotPresent);

    delete load;
    delete evict;
    delete evict2;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}



class TestCopyInputTask : public CopyInputTask {
public:
    bool is_success = false;
    bool is_error = false;
    int status_code;
    std::string error_message;
    std::shared_ptr<Allocation> workspace;

    TestCopyInputTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, char* input) : CopyInputTask(rm, cache, earliest, latest, input) {}

    void success(std::shared_ptr<Allocation> workspace) {
        is_success = true;
        this->workspace = workspace;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};


TEST_CASE("Copy Input", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask* copyinput = new TestCopyInputTask(rm, cache, 0, util::now() + 1000000000, input);
    copyinput->run(stream);
    while (!copyinput->is_complete());
    copyinput->process_completion();

    REQUIRE(copyinput->is_success);
    REQUIRE(!copyinput->is_error);

    free(input);
    delete copyinput;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}



class TestInferTask : public InferTask {
public:
    bool is_success = false;
    bool is_error = false;
    int status_code;
    std::string error_message;
    std::shared_ptr<Allocation> workspace;

    TestInferTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace) : InferTask(rm, cache, earliest, latest, workspace) {}

    void success(std::shared_ptr<Allocation> workspace) {
        is_success = true;
        this->workspace = workspace;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};

TEST_CASE("Input and Infer", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();


    TestLoadWeightsTask* loadweights = new TestLoadWeightsTask(rm, cache, 0, util::now()+1000000000);
    loadweights->run(stream);
    REQUIRE(!loadweights->is_error);
    
    while (!loadweights->is_complete());
    loadweights->process_completion();

    REQUIRE(loadweights->is_success);
    REQUIRE(!loadweights->is_error);


    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask* copyinput = new TestCopyInputTask(rm, cache, 0, util::now() + 1000000000, input);
    copyinput->run(stream);
    REQUIRE(!copyinput->is_error);
    
    while (!copyinput->is_complete());
    copyinput->process_completion();

    REQUIRE(copyinput->is_success);
    REQUIRE(!copyinput->is_error);

    TestInferTask* infer = new TestInferTask(rm, cache, 0, util::now() + 1000000000, copyinput->workspace);
    infer->run(stream);
    REQUIRE(!infer->is_error);

    while (!infer->is_complete());
    infer->process_completion();

    REQUIRE(infer->is_success);
    REQUIRE(!infer->is_error);

    free(input);
    delete copyinput;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}

TEST_CASE("Infer Without Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask* copyinput = new TestCopyInputTask(rm, cache, 0, util::now() + 1000000000, input);
    copyinput->run(stream);
    REQUIRE(!copyinput->is_error);
    
    while (!copyinput->is_complete());
    copyinput->process_completion();

    REQUIRE(copyinput->is_success);
    REQUIRE(!copyinput->is_error);

    TestInferTask* infer = new TestInferTask(rm, cache, 0, util::now() + 1000000000, copyinput->workspace);
    infer->run(stream);
    REQUIRE(!infer->is_success);
    REQUIRE(infer->is_error);
    REQUIRE(infer->status_code == actionErrorModelWeightsNotPresent);

    free(input);
    delete copyinput;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}




class TestCopyOutputTask : public CopyOutputTask {
public:
    bool is_success = false;
    bool is_error = false;
    int status_code;
    std::string error_message;

    TestCopyOutputTask(RuntimeModel* rm, PageCache* cache, uint64_t earliest, uint64_t latest, char* output, std::shared_ptr<Allocation> workspace) : CopyOutputTask(rm, cache, earliest, latest, output, workspace) {}

    void success() {
        is_success = true;
    }

    void error(int status_code, std::string message) {
        is_error = true;
        this->status_code = status_code;
        this->error_message = message;
    }

};



TEST_CASE("Input Infer and Output", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* cache = make_cache();
    cudaStream_t stream = make_stream();

    TestLoadWeightsTask* loadweights = new TestLoadWeightsTask(rm, cache, 0, util::now()+1000000000);
    loadweights->run(stream);
    REQUIRE(!loadweights->is_error);
    
    while (!loadweights->is_complete());
    loadweights->process_completion();

    REQUIRE(loadweights->is_success);
    REQUIRE(!loadweights->is_error);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask* copyinput = new TestCopyInputTask(rm, cache, 0, util::now() + 1000000000, input);
    copyinput->run(stream);
    REQUIRE(!copyinput->is_error);
    
    while (!copyinput->is_complete());
    copyinput->process_completion();

    REQUIRE(copyinput->is_success);
    REQUIRE(!copyinput->is_error);

    TestInferTask* infer = new TestInferTask(rm, cache, 0, util::now() + 1000000000, copyinput->workspace);
    infer->run(stream);
    REQUIRE(!infer->is_error);

    while (!infer->is_complete());
    infer->process_completion();

    REQUIRE(infer->is_success);
    REQUIRE(!infer->is_error);

    char* output = static_cast<char*>(malloc(model->output_size()));

    TestCopyOutputTask* copyoutput = new TestCopyOutputTask(rm, cache, 0, util::now() + 1000000000, output, infer->workspace);
    copyoutput->run(stream);
    REQUIRE(!copyoutput->is_error);

    while (!copyoutput->is_complete());
    copyoutput->process_completion();

    REQUIRE(copyoutput->is_success);
    REQUIRE(!copyoutput->is_error);


    free(input);
    free(output);
    delete loadweights;
    delete copyinput;
    delete infer;
    delete copyoutput;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(cache->baseptr) == cudaSuccess);
}
