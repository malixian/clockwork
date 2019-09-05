
#include <dlpack/dlpack.h>
#include <clockwork/manager.h>
#include <chrono>
#include <future>

namespace clockwork {

  extern std::mutex outLock;

  std::future<void> Manager::loadModel(const std::string& name, const std::string& source) {

    auto ret = std::make_shared<std::promise<void>>();

    this->model_manager_.lockModel(name);

    cpuExecLock_.lock();

    Task loadFromDisk("disk_load", [=] {
      this->model_manager_.loadModelFromDisk(name, source);
    }, name);
    Task* prev = disk_.addTask(loadFromDisk);

    Task createModel("cpu", [=]{
      this->model_manager_.instantiateModel(name);
      this->model_manager_.releaseModel(name);
      ret->set_value();
    }, prev, name);
    Task* next = cpu_.addTask(createModel);

    prev->setNextTask(next);

    cpuExecLock_.unlock();

    return ret->get_future();
  }

  std::future<void> Manager::loadToGPUAndInfer_(Model& model, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {
    auto ret = std::make_shared<std::promise<void>>();

    // lock on executors and enqueue the tasks
    this->execLock_.lock();
    model.status = ModelStatus::IN_USE;

    Task copyToDevice("load_to_device", [=, &model] {
      tvm::runtime::PackedFunc load = model.GetFunction("load_to_device");
      auto ctx = model.GetFunction("get_contig_context")();

      void* memaddr = load().operator void*();
      tvm::runtime::ManagedCUDADeviceAPI::Global()->ClaimOwnership(ctx, memaddr, model.name);
    }, model.name);
    Task* prev = this->load_to_device_.addTask(copyToDevice);

    Task uploadInput("input", [=, &model] {
      tvm::runtime::PackedFunc set_input = model.GetFunction("set_input");
      set_input(inputName, input);
    }, prev, model.name);
    Task* next = upload_inputs_.addTask(uploadInput);

    prev->setNextTask(next);
    prev = next;

    Task run("run", [=, &model] {
      tvm::runtime::PackedFunc run = model.GetFunction("run");
      run();
    }, prev, model.name);
    next = gpu_.addTask(run);

    prev->setNextTask(next);
    prev = next;

    Task getOutput("output", [=, &model] {
      tvm::runtime::PackedFunc get_output = model.GetFunction("get_output");
      get_output(outIndex, output);
    }, prev, model.name);
    next = this->d2h_pcie_.addTask(getOutput);

    prev->setNextTask(next);
    prev = next;

    Task finish("post_output", [=, &model] {
      // update model status
      model.status = ModelStatus::READY;
      model.last_use = std::chrono::high_resolution_clock::now();

      // release model for use
      this->model_manager_.releaseModel(model.name);

      ret->set_value();
    }, prev, model.name);
    next = this->out_proc_.addTask(finish);

    prev->setNextTask(next);

    this->execLock_.unlock();

    return ret->get_future();
  }

  std::future<void> Manager::infer_(Model& model, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {
    auto ret = std::make_shared<std::promise<void>>();

    // lock on executors and enqueue the tasks
    this->execLock_.lock();
    model.status = ModelStatus::IN_USE;

    Task uploadInput("input", [=, &model] {
      tvm::runtime::PackedFunc set_input = model.GetFunction("set_input");
      set_input(inputName, input);
    }, model.name);
    Task* prev = upload_inputs_.addTask(uploadInput);

    Task run("run", [=, &model] {
      tvm::runtime::PackedFunc run = model.GetFunction("run");
      run();
    }, prev, model.name);
    Task* next = gpu_.addTask(run);

    prev->setNextTask(next);
    prev = next;

    Task getOutput("output", [=, &model] {
      tvm::runtime::PackedFunc get_output = model.GetFunction("get_output");
      get_output(outIndex, output);
    }, prev, model.name);
    next = this->d2h_pcie_.addTask(getOutput);

    prev->setNextTask(next);
    prev = next;

    Task finish("post_output", [=, &model] {
      // update model status
      model.status = ModelStatus::READY;
      model.last_use = std::chrono::high_resolution_clock::now();

      // release model for use
      this->model_manager_.releaseModel(model.name);

      ret->set_value();
    }, prev, model.name);
    next = this->out_proc_.addTask(finish);

    prev->setNextTask(next);

    this->execLock_.unlock();

    return ret->get_future();
  }

}  // namespace clockwork
