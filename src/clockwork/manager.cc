
#include <dlpack/dlpack.h>
#include <clockwork/runtime.h>
#include <clockwork/manager.h>
#include <clockwork/model_manager.h>
#include <chrono>
#include <future>

namespace clockwork {

    extern std::mutex outLock;


    Manager::Manager(int managedMemorySize, Runtime* runtime) : pendingGPUUploads_(0), runtime_(runtime) {
      model_manager_ = new ModelManager();
      memory_manager_ = new ClockworkMemoryManager(managedMemorySize, model_manager_);
      tvm::runtime::ManagedCUDADeviceAPI::Global()->SetDataspaceManager(memory_manager_);
    }


    std::future<void> Manager::loadModel(const std::string& name, const std::string& source) {

      auto ret = std::make_shared<std::promise<void>>();

      this->model_manager_->lockModel(name);

      RequestBuilder* b = runtime_->newRequest();

      b->addTask(TaskType::Disk, [=] {
        this->model_manager_->loadModelFromDisk(name, source);
      });

      b->addTask(TaskType::CPU, [=]{
        this->model_manager_->instantiateModel(name);
        this->model_manager_->releaseModel(name);
        ret->set_value();
      });

      b->submit();

      return ret->get_future();
    }

    std::future<void> Manager::infer(const std::string& name, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {

      while (pendingGPUUploads_.load() > 0) {
        usleep(1000);
      } // wait until all evictions have passed

      auto& model = model_manager_->getModel(name);

      model.last_use = std::chrono::high_resolution_clock::now();

      if (rand() % 100 < kEvictionRate) {
        model.status = ModelStatus::EVICTED;
        model.GetFunction("evicted")();
        this->model_manager_->insertFauxEviction(name);
        return loadToGPUAndInfer_(model, inputName, input, outIndex, output);
      } else if (model.status == ModelStatus::READY) {
        return infer_(model, inputName, input, outIndex, output);
      } else if (model.status == ModelStatus::EVICTED) {
        return loadToGPUAndInfer_(model, inputName, input, outIndex, output);
      } else {
        CHECK(false) << "We've gotten the lock for a model while it was in use.";
      }
    }

  std::future<void> Manager::loadToGPUAndInfer_(Model& model, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {
    auto ret = std::make_shared<std::promise<void>>();

    model.status = ModelStatus::IN_USE;

    this->pendingGPUUploads_++;

    RequestBuilder* b = runtime_->newRequest();

    b->addTask(TaskType::PCIe_H2D_Weights, [=, &model] {
      tvm::runtime::PackedFunc load = model.GetFunction("load_to_device");
      TVMContext ctx = model.GetFunction("get_contig_context")();

      void* memaddr = load().operator void*();
      this->pendingGPUUploads_--;
      memory_manager_->ClaimOwnership(ctx.device_id, memaddr, model.name);
    });

    b->addTask(TaskType::PCIe_H2D_Inputs, [=, &model] {
      tvm::runtime::PackedFunc set_input = model.GetFunction("set_input");
      set_input(inputName, input);
    });

    b->addTask(TaskType::GPU, [=, &model] {
      tvm::runtime::PackedFunc run = model.GetFunction("run");
      run();
    });

    b->addTask(TaskType::PCIe_D2H_Output, [=, &model] {
      tvm::runtime::PackedFunc get_output = model.GetFunction("get_output");
      get_output(outIndex, output);
    });

    b->addTask(TaskType::Sync, [=, &model] {
      // update model status
      model.status = ModelStatus::READY;
      model.last_use = std::chrono::high_resolution_clock::now();

      // release model for use
      this->model_manager_->releaseModel(model.name);

      ret->set_value();
    });

    b->submit();

    return ret->get_future();
  }

  std::future<void> Manager::infer_(Model& model, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {
    auto ret = std::make_shared<std::promise<void>>();

    RequestBuilder* b = runtime_->newRequest();

    b->addTask(TaskType::PCIe_H2D_Inputs, [=, &model] {
      tvm::runtime::PackedFunc set_input = model.GetFunction("set_input");
      set_input(inputName, input);
    });

    b->addTask(TaskType::GPU, [=, &model] {
      tvm::runtime::PackedFunc run = model.GetFunction("run");
      run();
    });

    b->addTask(TaskType::PCIe_D2H_Output, [=, &model] {
      tvm::runtime::PackedFunc get_output = model.GetFunction("get_output");
      get_output(outIndex, output);
    });

    b->addTask(TaskType::Sync, [=, &model] {
      // update model status
      model.status = ModelStatus::READY;
      model.last_use = std::chrono::high_resolution_clock::now();

      // release model for use
      this->model_manager_->releaseModel(model.name);

      ret->set_value();
    });

    b->submit();

    return ret->get_future();
  }

}  // namespace clockwork
