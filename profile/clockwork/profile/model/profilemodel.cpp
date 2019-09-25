#include <shared_mutex>
#include <algorithm>
#include <chrono>
#include <thread>
#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/so.h"
#include "clockwork/model/cuda.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include <sys/mman.h>

using namespace clockwork;

model::Model* duplicate(model::Model* model, bool duplicate_weights) {
    Memfile so_memfile = Memfile::readFrom(model->so_memfile.filename);

    std::string serialized_spec = model->serialized_spec;

    void* weights_pinned_host_memory;
    if (duplicate_weights) {
        weights_pinned_host_memory = malloc(model->weights_size);
        // cudaError_t status = cudaHostRegister(weights_pinned_host_memory, model->weights_size, cudaHostRegisterDefault);
        // REQUIRE(status == cudaSuccess);
        REQUIRE(mlock(weights_pinned_host_memory, model->weights_size) == 0);
        //CUDA_CALL(cudaMallocHost(&weights_pinned_host_memory, model->weights_size));
        std::memcpy(weights_pinned_host_memory, model->weights_pinned_host_memory, model->weights_size);
    } else {
        weights_pinned_host_memory = model->weights_pinned_host_memory;
    }

    return new model::Model(so_memfile, serialized_spec, model->weights_size, static_cast<char*>(weights_pinned_host_memory));
}

class Experiment {
public:
    std::atomic_int progress;
    std::shared_mutex shared;
    std::mutex copy_weights_mutex, copy_inputs_mutex, exec_mutex, copy_output_mutex;
    cudaStream_t copy_weights_stream, copy_inputs_stream, exec_stream, copy_output_stream;

    Experiment() : progress(0) {
        cudaError_t status;
        status = cudaStreamCreate(&copy_inputs_stream);
        REQUIRE(status == cudaSuccess);
        status = cudaStreamCreate(&copy_weights_stream);
        REQUIRE(status == cudaSuccess);
        status = cudaStreamCreate(&exec_stream);
        REQUIRE(status == cudaSuccess);
        status = cudaStreamCreate(&copy_output_stream);
        REQUIRE(status == cudaSuccess);
    }
};

struct Measurement {
    std::vector<float> cuda;
    std::vector<uint64_t> host;
    Measurement(std::vector<cudaEvent_t> &events, std::vector<std::chrono::high_resolution_clock::time_point> &timestamps) : cuda(events.size()/2), host(timestamps.size()/2) {
        cudaError_t status;
        for (int i = 1; i < events.size(); i += 1) {
            float duration;
            status = cudaEventElapsedTime(&duration, events[i-1], events[i]);
            REQUIRE(status == cudaSuccess);
            cuda[i/2] = duration;
        }
        for (int i = 1; i < timestamps.size(); i += 2) {
            host[i/2] = util::nanos(timestamps[i]) - util::nanos(timestamps[i-1]);
        } 
    }
};

struct Series {
    std::vector<std::vector<uint64_t>> data;

    Series(std::vector<Measurement> &measurements) {
        data.resize(measurements[0].cuda.size() + measurements[0].host.size());

        for (unsigned i = 0; i < data.size(); i++) {
            data[i].resize(measurements.size());
        }

        for (unsigned i = 0; i < measurements.size(); i++) {
            unsigned next = 0;
            for (float &f : measurements[i].cuda) {
                data[next++][i] = 1000000 * f;
            }
            for (uint64_t &x : measurements[i].host) {
                data[next++][i] = x;
            }
        }
    }

    std::vector<uint64_t> medians() {
        std::vector<uint64_t> medians(data.size());
        for (unsigned i = 0; i < data.size(); i++) {
            std::vector<uint64_t> series(data[i]);
            std::sort(series.begin(), series.end());
            medians[i] = series[series.size()/2]; 
        }
        return medians;
    }

    std::vector<uint64_t> percentiles(float p) {
        std::vector<uint64_t> ps(data.size());
        for (unsigned i = 0; i < data.size(); i++) {
            std::vector<uint64_t> series(data[i]);
            std::sort(series.begin(), series.end());
            ps[i] = series[(int) ((series.size()-1)*p)]; 
        }
        return ps;
    }

};

class ClosedLoopModelExec {
public:
    std::atomic_int iterations;
    std::thread thread;
    std::atomic_bool alive;
    PageCache* cache;
    Experiment* experiment;
    std::vector<model::Model*> models;
    std::string input;


    std::vector<Measurement> measurements;

    void run() {
        util::initializeCudaStream();

        for (model::Model* model : models) {
            model->instantiate_model_on_host();
        }


        cudaError_t status;

        std::vector<cudaEvent_t> events(8);
        for (unsigned i = 0; i < events.size(); i++) {
            status = cudaEventCreate(&events[i]);
            REQUIRE(status == cudaSuccess);            
        }

        while (alive) {
            model::Model* model = models[rand() % models.size()];

            std::vector<std::chrono::high_resolution_clock::time_point> timestamps;
            timestamps.reserve(12);

            experiment->shared.lock();
            timestamps.push_back(util::hrt());
            model->instantiate_model_on_device();
            experiment->shared.unlock();
            timestamps.push_back(util::hrt());

            std::shared_ptr<Allocation> weights = cache->alloc(model->num_weights_pages(cache->page_size), []{});

            experiment->shared.lock_shared();

            experiment->copy_weights_mutex.lock();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[0], experiment->copy_weights_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_weights_to_device(weights->page_pointers, experiment->copy_weights_stream);
            status = cudaEventRecord(events[1], experiment->copy_weights_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_weights_mutex.unlock();

            status = cudaEventSynchronize(events[1]);
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            std::shared_ptr<Allocation> workspace = cache->alloc(model->num_workspace_pages(cache->page_size), []{});

            experiment->copy_inputs_mutex.lock();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[2], experiment->copy_inputs_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_input_to_device(input.data(), workspace->page_pointers, experiment->copy_inputs_stream);
            status = cudaEventRecord(events[3], experiment->copy_inputs_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_inputs_mutex.unlock();

            status = cudaEventSynchronize(events[3]);
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            experiment->exec_mutex.lock();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[4], experiment->exec_stream);
            REQUIRE(status == cudaSuccess);
            model->call(weights->page_pointers, workspace->page_pointers, experiment->exec_stream);
            status = cudaEventRecord(events[5], experiment->exec_stream);
            REQUIRE(status == cudaSuccess);
            experiment->exec_mutex.unlock();

            REQUIRE(status == cudaSuccess);

            status = cudaEventSynchronize(events[5]);
            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            char output[model->output_size()];

            experiment->copy_output_mutex.lock();
            timestamps.push_back(util::hrt());
            status = cudaEventRecord(events[6], experiment->copy_output_stream);
            REQUIRE(status == cudaSuccess);
            model->transfer_output_from_device(output, workspace->page_pointers, experiment->copy_output_stream);
            status = cudaEventRecord(events[7], experiment->copy_output_stream);
            REQUIRE(status == cudaSuccess);
            experiment->copy_output_mutex.unlock();

            REQUIRE(status == cudaSuccess);

            status = cudaEventSynchronize(events[7]);

            experiment->shared.unlock_shared();

            REQUIRE(status == cudaSuccess);
            timestamps.push_back(util::hrt());

            cache->unlock(workspace);
            cache->unlock(weights);
            cache->free(workspace);
            cache->free(weights);


            experiment->shared.lock();
            timestamps.push_back(util::hrt());
            model->uninstantiate_model_on_device(); 
            experiment->shared.unlock();
            timestamps.push_back(util::hrt());

            iterations++;
            experiment->progress++;

            measurements.push_back(Measurement(events, timestamps));
        }

        for (model::Model* model : models) {
            model->uninstantiate_model_on_host();
            delete model;
        }

    }

    ClosedLoopModelExec(int i, Experiment* experiment, PageCache* cache, std::vector<model::Model*> models, std::string input) :
            experiment(experiment), cache(cache), models(models), alive(true), input(input), iterations(0) {
        util::set_core((i+7) % util::get_num_cores());
        util::setCurrentThreadMaxPriority();
    }

    void start() {
        thread = std::thread(&ClosedLoopModelExec::run, this);
    }

    void stop(bool awaitCompletion) {
        alive = false;
        if (awaitCompletion) {
            join();
        }
    }

    void join() {
        thread.join();
    }

    void awaitIterations(int iterations) {
        while (this->iterations.load() < iterations) {}
    }

};

model::Model* load_model_from_disk(std::string model_path) {
    std::string so_filename = model_path + ".so";
    std::string clockwork_filename = model_path + ".clockwork";
    std::string params_filename = model_path + ".clockwork_params";
    return model::Model::loadFromDisk(so_filename, clockwork_filename, params_filename);    
}

void get_model_inputs_and_outputs(std::string model_path, std::string &input, std::string &output) {
    std::string input_filename = model_path + ".input";
    std::string output_filename = model_path + ".output";
    clockwork::util::readFileAsString(input_filename, input);
    clockwork::util::readFileAsString(output_filename, output);
}

PageCache* make_cache(size_t size, size_t page_size) { 
    void* baseptr;
    cudaError_t status = cudaMalloc(&baseptr, size);
    REQUIRE(status == cudaSuccess);
    return new PageCache(static_cast<char*>(baseptr), size, page_size);
}

void warmup() {
    util::initializeCudaStream();

    std::string model_name = "resnet50";
    std::string model_path = clockwork::util::get_example_model("resnet50_tesla-m40_batchsize1");

    size_t page_size = 16 * 1024L * 1024L;
    size_t cache_size = 512L * page_size;
    PageCache* cache = make_cache(cache_size, page_size);

    model::Model* model = load_model_from_disk(model_path);

    std::string input, expected_output;
    get_model_inputs_and_outputs(model_path, input, expected_output);

    cudaError_t status;

    model->instantiate_model_on_host();
    model->instantiate_model_on_device();
    
    std::shared_ptr<Allocation> weights = cache->alloc(model->num_weights_pages(cache->page_size), []{});
    model->transfer_weights_to_device(weights->page_pointers, util::Stream());

    std::shared_ptr<Allocation> workspace = cache->alloc(model->num_workspace_pages(cache->page_size), []{});
    model->transfer_input_to_device(input.data(), workspace->page_pointers, util::Stream());
    
    model->call(weights->page_pointers, workspace->page_pointers, util::Stream());

    char output[model->output_size()];
    model->transfer_output_from_device(output, workspace->page_pointers, util::Stream());

    status = cudaStreamSynchronize(util::Stream());
    REQUIRE(status == cudaSuccess);

    cache->unlock(workspace);
    cache->free(workspace);

    cache->unlock(weights);
    cache->free(weights);

    model->uninstantiate_model_on_device();
    model->uninstantiate_model_on_host();

    status = cudaFreeHost(model->weights_pinned_host_memory);
    REQUIRE(status == cudaSuccess);

    status = cudaFree(cache->baseptr);
    REQUIRE(status == cudaSuccess);

    delete model;
    delete cache;
}

TEST_CASE("Warmup works", "[profile] [warmup]") {
    for (unsigned i = 0; i < 3; i++) {
        warmup();
    }
}

TEST_CASE("Profile resnet50 1 thread", "[profile] [resnet50] [e2e]") {
    util::setCudaFlags();
    for (unsigned i = 0; i < 3; i++) {
        warmup();
    }
    
    util::initializeCudaStream();

    std::string model_name = "resnet50";
    std::string model_path = clockwork::util::get_example_model("resnet50_tesla-m40_batchsize1");

    size_t page_size = 16 * 1024L * 1024L;
    size_t cache_size = 512L * page_size;
    PageCache* cache = make_cache(cache_size, page_size);

    Experiment* experiment = new Experiment();

    int num_execs = 10;
    int models_per_exec = 100;
    bool duplicate_weights = true;
    std::vector<ClosedLoopModelExec*> execs;

    model::Model* model;
    std::string input, output;
    get_model_inputs_and_outputs(model_path, input, output);

    for (unsigned i = 0; i < num_execs; i++) {
        std::vector<model::Model*> models;
        for (unsigned j = 0; j < models_per_exec; j++) {
            if (i == 0 && j == 0) {
                model = load_model_from_disk(model_path);
            } else {
                model = duplicate(model, duplicate_weights);
            }
            models.push_back(model);

            unsigned progress = (100 * ((i * models_per_exec) + j)) / (models_per_exec*num_execs);
            std::cout << "Creating model: " << ((i * models_per_exec) + j) << " (" << progress << "%) \n";
            std::cout.flush();
        }

        ClosedLoopModelExec* exec = new ClosedLoopModelExec(i,
            experiment, cache, models, input);

        execs.push_back(exec);
    }

    for (ClosedLoopModelExec* exec : execs) {
        exec->start();
    }

    std::cout << "Exec creation completed, awaiting termination" << std::endl;

    int iterations = 10000;
    int progress;
    while ((progress = experiment->progress.load()) < iterations) {
        std::cout << progress << " (" << ((progress * 100) / iterations) << "%) \r";
        std::cout.flush();
    }
    for (int i = 0; i < execs.size(); i++) {
        execs[i]->stop(false);
    }
    for (int i = 0; i < execs.size(); i++) {
        execs[i]->join();
    }

    std::vector<Measurement> measurements;
    for (int i = 0; i < execs.size(); i++) {
        measurements.insert(measurements.end(), execs[i]->measurements.begin(), execs[i]->measurements.end());
    }

    std::vector<std::string> series_names = {
        "cWeights", "cInputs", "cKernel", "cOutputs",
        "hModuleLoad", "hWeights", "hInputs", "hKernel", "hOutputs", "hModuleUnload"
    };


    Series series(measurements);

    std::vector<uint64_t> medians = series.percentiles(0.5);
    std::vector<uint64_t> p99 = series.percentiles(0.99);
    std::vector<uint64_t> p999 = series.percentiles(0.999);
    std::vector<uint64_t> p9999 = series.percentiles(0.9999);
    std::vector<uint64_t> p99999 = series.percentiles(0.99999);

    for (unsigned i = 0; i < series_names.size(); i++) {
        std::cout << series_names[i] << ":  median " << medians[i];
        std::cout << "   p99 +";
        printf("%.2f", 100 * (((float) p99[i]) / ((float) medians[i]) - 1));
        std::cout << "   p99.9 +";
        printf("%.2f", 100 * (((float) p999[i]) / ((float) medians[i]) - 1));
        std::cout << "%   p99.99 +";
        printf("%.2f", 100 * (((float) p9999[i]) / ((float) medians[i]) - 1));
        std::cout << "%   p99.999 +";
        printf("%.2f", 100 * (((float) p99999[i]) / ((float) medians[i]) - 1));
        std::cout << "%" << std::endl;
    }

    for (int i = 0; i < execs.size(); i++) {
        delete execs[i];
    }

    delete experiment;
    delete cache;
}