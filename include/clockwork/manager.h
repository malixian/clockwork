/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef CLOCKWORK_MULTITENANT_MANAGER_H_
#define CLOCKWORK_MULTITENANT_MANAGER_H_

#include <clockwork/model_manager.h>
#include <clockwork/decoupledruntime.h>
#include <clockwork/greedyruntime.h>
#include <clockwork/threadpoolruntime.h>
#include <clockwork/clockworkruntime.h>
#include <tvm/runtime/managed_cuda_device_api.h>
#include <climits>
#include <cstdlib>
#include <map>
#include <unistd.h>

namespace clockwork {

  class Manager {
  public:
    Manager() : pendingGPUUploads_(0) {
      runtime_ = newDecoupledRuntime(1, 1, 1, 1, 8);
    }

    std::future<void> loadModel(const std::string& name, const std::string& source);

    std::future<void> infer(const std::string& name, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output) {

      while (pendingGPUUploads_.load() > 0) {
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
    Runtime* runtime_;
    std::atomic<unsigned> pendingGPUUploads_;

  };

}  // namespace clockwork

#endif  // CLOCKWORK_MULTITENANT_MANAGER_H_
