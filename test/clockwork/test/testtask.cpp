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
    std::shared_ptr<Allocation> workspace;

    TestCopyInputTask(MemoryManager* cache, int model_id, uint64_t earliest, uint64_t latest, size_t input_size, char* input) : CopyInputTask(cache, model_id, earliest, latest, input_size, input), workspace(nullptr) {}

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

    void success(RuntimeModel* rm, std::shared_ptr<Allocation> workspace) {
        is_success = true;
        this->rm = rm;
        this->workspace = workspace;
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

    TestInferTask(RuntimeModel* rm, MemoryManager* cache, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace) : ExecTask(rm, cache, earliest, latest, workspace) {}

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

    TestCopyOutputTask(RuntimeModel* rm, MemoryManager* manager, uint64_t earliest, uint64_t latest, std::shared_ptr<Allocation> workspace) : CopyOutputTask(rm, manager, earliest, latest, workspace) {}

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

MemoryManager* make_manager(PageCache* cache) {
    MemoryManager* manager = new MemoryManager();
    manager->weights_cache = cache;
    manager->workspace_cache = cache;
    manager->models = new ModelStore();
    return manager;
}

TEST_CASE("Load Model From Disk", "[task] [loadmodel]") {
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);

    int model_id = 0;
    std::string model_path = clockwork::util::get_example_model();

    TestLoadModelFromDiskTask* task = 
        new TestLoadModelFromDiskTask(cache, model_id, model_path, util::now(), util::now()+1000000000);

    task->run(stream);

    INFO(task->status_code << ": " << task->error_message);
    REQUIRE(!task->is_error);
    REQUIRE(task->is_success);

    delete task;
    delete pagecache;
    delete cache;
}

TEST_CASE("Load Non-Existent Model From Disk", "[task] [loadmodel]") {
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);

    int model_id = 0;
    std::string model_path = clockwork::util::get_example_model() + "bad";

    TestLoadModelFromDiskTask* task = 
        new TestLoadModelFromDiskTask(cache, model_id, model_path, util::now(), util::now()+1000000000);

    task->run(stream);

    INFO(task->status_code << ": " << task->error_message);
    REQUIRE(task->is_error);
    REQUIRE(!task->is_success);

    delete task;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
    delete cache;
}

TEST_CASE("Load Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    uint64_t now = util::now();
    TestLoadWeightsTask* task = new TestLoadWeightsTask(cache, 0, now, now+1000000000);

    REQUIRE(task->eligible() == now);


    task->run(stream);

    REQUIRE(!task->is_complete());

    while (!task->is_complete());

    task->process_completion();

    REQUIRE(task->is_success);
    REQUIRE(!task->is_error);

    delete task;
    delete model;
    delete rm;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
    delete cache;
}

TEST_CASE("Load Weights Nonexistent Model", "[task]") {
    Model* model = make_model();
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);

    uint64_t now = util::now();
    TestLoadWeightsTask* task = new TestLoadWeightsTask(cache, 0, now, now+1000000000);

    REQUIRE(task->eligible() == now);


    task->run(stream);
    REQUIRE(task->is_error);
    REQUIRE(!task->is_success);
    REQUIRE(task->status_code == actionErrorUnknownModel);

    delete task;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
    delete cache;
}

TEST_CASE("Load Weights Earliest", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask* task = new TestLoadWeightsTask(cache, 0, now+1000000000, now+1000000000);

    task->run(stream);

    REQUIRE(task->is_complete());
    REQUIRE(!task->is_success);
    REQUIRE(task->is_error);
    REQUIRE(task->status_code == actionErrorRuntimeError);

    delete task;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Load Weights Latest", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask* task = new TestLoadWeightsTask(cache, 0, 0, now - 1000000);

    task->run(stream);

    REQUIRE(task->is_complete());
    REQUIRE(!task->is_success);
    REQUIRE(task->is_error);
    REQUIRE(task->status_code == actionErrorCouldNotStartInTime);

    delete task;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Load Weights Insufficient Cache", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache(16 * 1024 * 1024, 16 * 1024 * 1024);
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask* task = new TestLoadWeightsTask(cache, 0, 0, now + 1000000000L);

    task->run(stream);

    REQUIRE(task->is_complete());
    REQUIRE(!task->is_success);
    REQUIRE(task->is_error);
    REQUIRE(task->status_code == actionErrorRuntimeError);

    delete task;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Load Weights Version Update", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    uint64_t now = util::now();

    TestLoadWeightsTask* task = new TestLoadWeightsTask(cache, 0, 0, now + 1000000000L);

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
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Double Load Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    // Load weights 1
    TestLoadWeightsTask* load1 = new TestLoadWeightsTask(cache, 0, 0, util::now() + 1000000000L);
    load1->run(stream);

    rm->lock();

    int invalid_version = rm->version;
    std::shared_ptr<Allocation> invalid_weights = rm->weights;

    rm->unlock();

    // Load weights 2
    TestLoadWeightsTask* load2 = new TestLoadWeightsTask(cache, 0, 0, util::now() + 1000000000L);
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
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}



TEST_CASE("Evict Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    // Load weights
    uint64_t now = util::now();
    TestLoadWeightsTask* load = new TestLoadWeightsTask(cache, 0, 0, util::now() + 1000000000L);
    load->run(stream);
    while (!load->is_complete());
    load->process_completion();
    
    REQUIRE(load->is_success);
    REQUIRE(!load->is_error);

    // Now evict them
    TestEvictWeightsTask* evict = new TestEvictWeightsTask(cache, 0, 0, util::now() + 1000000000);
    evict->run(stream);

    REQUIRE(evict->is_success);
    REQUIRE(!evict->is_error);

    delete load;
    delete evict;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Evict Non-Existent Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    // Evict weights
    TestEvictWeightsTask* evict = new TestEvictWeightsTask(cache, 0, 0, util::now() + 1000000000);
    evict->run(stream);

    REQUIRE(!evict->is_success);
    REQUIRE(evict->is_error);
    REQUIRE(evict->is_error);
    REQUIRE(evict->status_code == actionErrorModelWeightsNotPresent);

    delete evict;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Double Evict", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    // Load weights
    uint64_t now = util::now();
    TestLoadWeightsTask* load = new TestLoadWeightsTask(cache, 0, 0, util::now() + 1000000000L);
    load->run(stream);
    while (!load->is_complete());
    load->process_completion();
    
    REQUIRE(load->is_success);
    REQUIRE(!load->is_error);

    // Now evict them
    TestEvictWeightsTask* evict = new TestEvictWeightsTask(cache, 0, 0, util::now() + 1000000000);
    evict->run(stream);

    REQUIRE(evict->is_success);
    REQUIRE(!evict->is_error);

    // Now evict them
    TestEvictWeightsTask* evict2 = new TestEvictWeightsTask(cache, 0, 0, util::now() + 1000000000);
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
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Evict Weights Nonexistent Model", "[task]") {
    Model* model = make_model();
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);

    uint64_t now = util::now();
    TestEvictWeightsTask* task = new TestEvictWeightsTask(cache, 0, now, now+1000000000);

    REQUIRE(task->eligible() == now);


    task->run(stream);
    REQUIRE(task->is_error);
    REQUIRE(!task->is_success);
    REQUIRE(task->status_code == actionErrorUnknownModel);

    delete task;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
    delete cache;
}





TEST_CASE("Copy Input", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask* copyinput = new TestCopyInputTask(cache, 0, 0, util::now() + 1000000000, model->input_size(), input);
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
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Copy Input Wrong Size", "[task] [wrongsize]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    char* input = static_cast<char*>(malloc(10));

    TestCopyInputTask* copyinput = new TestCopyInputTask(cache, 0, 0, util::now() + 1000000000, 10, input);
    copyinput->run(stream);
    while (!copyinput->is_complete());

    REQUIRE(!copyinput->is_success);
    REQUIRE(copyinput->is_error);
    REQUIRE(copyinput->status_code == actionErrorInvalidInput);

    free(input);
    delete copyinput;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}

TEST_CASE("Copy Input Nonexistent Model", "[task]") {
    Model* model = make_model();
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);

    char* input = static_cast<char*>(malloc(model->input_size()));

    uint64_t now = util::now();
    TestCopyInputTask* task = new TestCopyInputTask(cache, 0, now, util::now() + 1000000000, model->input_size(), input);

    REQUIRE(task->eligible() == now);


    task->run(stream);
    REQUIRE(task->is_error);
    REQUIRE(!task->is_success);
    REQUIRE(task->status_code == actionErrorUnknownModel);

    delete task;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
    delete cache;
}


TEST_CASE("Input and Infer", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);


    TestLoadWeightsTask* loadweights = new TestLoadWeightsTask(cache, 0, 0, util::now()+1000000000);
    loadweights->run(stream);
    REQUIRE(!loadweights->is_error);
    
    while (!loadweights->is_complete());
    loadweights->process_completion();

    REQUIRE(loadweights->is_success);
    REQUIRE(!loadweights->is_error);


    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask* copyinput = new TestCopyInputTask(cache, 0, 0, util::now() + 1000000000, model->input_size(), input);
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
    delete infer;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
}

TEST_CASE("Infer Without Weights", "[task]") {
    Model* model = make_model();
    RuntimeModel* rm = new RuntimeModel(model);
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);
    cache->models->put(0, rm);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask* copyinput = new TestCopyInputTask(cache, 0, 0, util::now() + 1000000000, model->input_size(), input);
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
    delete infer;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}






TEST_CASE("Input Infer and Output", "[task]") {
    PageCache* pagecache = make_cache();
    cudaStream_t stream = make_stream();
    MemoryManager* cache = make_manager(pagecache);

    std::string model_path = clockwork::util::get_example_model();

    TestLoadModelFromDiskTask* loadmodel = 
        new TestLoadModelFromDiskTask(cache, 0, model_path, util::now(), util::now()+1000000000);

    loadmodel->run(stream);
    REQUIRE(loadmodel->is_success);
    REQUIRE(!loadmodel->is_error);

    RuntimeModel* rm = cache->models->get(0);
    REQUIRE(rm != nullptr);
    model::Model* model = rm->model;

    TestLoadWeightsTask* loadweights = new TestLoadWeightsTask(cache, 0, 0, util::now()+1000000000);
    loadweights->run(stream);
    REQUIRE(!loadweights->is_error);
    
    while (!loadweights->is_complete());
    loadweights->process_completion();

    REQUIRE(loadweights->is_success);
    REQUIRE(!loadweights->is_error);

    char* input = static_cast<char*>(malloc(model->input_size()));

    TestCopyInputTask* copyinput = new TestCopyInputTask(cache, 0, 0, util::now() + 1000000000, model->input_size(), input);
    copyinput->run(stream);
    REQUIRE(!copyinput->is_error);
    
    while (!copyinput->is_complete());
    copyinput->process_completion();

    REQUIRE(copyinput->is_success);
    REQUIRE(!copyinput->is_error);
    REQUIRE(copyinput->workspace != nullptr);

    TestInferTask* infer = new TestInferTask(rm, cache, 0, util::now() + 1000000000, copyinput->workspace);
    infer->run(stream);
    REQUIRE(!infer->is_error);

    while (!infer->is_complete());
    infer->process_completion();

    REQUIRE(infer->is_success);
    REQUIRE(!infer->is_error);

    TestCopyOutputTask* copyoutput = new TestCopyOutputTask(rm, cache, 0, util::now() + 1000000000, copyinput->workspace);
    copyoutput->run(stream);
    REQUIRE(!copyoutput->is_error);

    while (!copyoutput->is_complete());
    copyoutput->process_completion();

    REQUIRE(copyoutput->is_success);
    REQUIRE(!copyoutput->is_error);


    free(input);
    delete loadmodel;
    delete loadweights;
    delete copyinput;
    delete infer;
    delete copyoutput;
    delete cache;
    delete rm;
    delete model;
    REQUIRE(cudaFree(pagecache->baseptr) == cudaSuccess);
    delete pagecache;
}
