/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef CLOCKWORK_MULTITENANT_MANAGER_H_
#define CLOCKWORK_MULTITENANT_MANAGER_H_

#include <clockwork/executor.h>
#include <clockwork/model_manager.h>
#include <tvm/runtime/managed_cuda_device_api.h>
#include <climits>
#include <cstdlib>
#include <map>
#include <unistd.h>

namespace clockwork {

  class Manager {
  public:
    Manager() : disk_(false),
                cpu_(true),
                load_to_device_(false),
                upload_inputs_(false),
                gpu_(false, 8),
                d2h_pcie_(false),
                out_proc_(true) {}

    std::future<void> loadModel(const std::string& name, const std::string& source);

    std::future<void> infer(const std::string& name, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {

      while (!load_to_device_.empty()) {
        usleep(1000);
      } // wait until all evictions have passed

      auto& model = model_manager_.getModel(name);

      model.last_use = std::chrono::high_resolution_clock::now();

      if (rand() % 100 < kEvictionRate) {
        model.status = ModelStatus::EVICTED;
        model.GetFunction("evicted")();
        this->model_manager_.insertFauxEviction(name);
        return loadToGPUAndInfer_(model, inputName, input, outIndex, output);
      } else if (model.status == ModelStatus::READY) {
        return infer_(model, inputName, input, outIndex, output);
      } else if (model.status == ModelStatus::EVICTED) {
        return loadToGPUAndInfer_(model, inputName, input, outIndex, output);
      } else {
        CHECK(false) << "We've gotten the lock for a model while it was in use.";
      }
    }

  private:

    std::future<void> loadToGPUAndInfer_(Model& model, const std::string& inputName,
                  DLTensor* input, int outIndex, DLTensor* output);

    std::future<void> infer_(Model& model, const std::string& inputName, DLTensor* input,
                  int outIndex, DLTensor* output);

    ModelManager model_manager_;

    // model loaded to cpu in separate pipeline
    std::mutex cpuExecLock_;
    Executor disk_; // loading model files from disk
    Executor cpu_; // preprocessing model

    // Executors for each resource and a lock to ensure correct ordering
    std::mutex execLock_;
    // two executors for h2d_pcie
    Executor load_to_device_; // upload model params, setup functions, etc..
    Executor upload_inputs_; // copy input for inference

    Executor gpu_; // execute on gpu
    Executor d2h_pcie_; // copy outputs to host
    Executor out_proc_; // any post processing needed after output is copied back
  };

}  // namespace clockwork

#endif  // CLOCKWORK_MULTITENANT_MANAGER_H_
