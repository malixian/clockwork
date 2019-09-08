/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef CLOCKWORK_MULTITENANT_MANAGER_H_
#define CLOCKWORK_MULTITENANT_MANAGER_H_

#include <tvm/runtime/managed_cuda_device_api.h>
#include <clockwork/clockwork.h>
#include <climits>
#include <cstdlib>
#include <map>
#include <unistd.h>
#include <future>

namespace clockwork {

  class ModelManager;
  class ClockworkMemoryManager;
  class Model;

  class Manager {
  private:
    ModelManager* model_manager_;
    ClockworkMemoryManager* memory_manager_; 
    Runtime* runtime_;
    std::atomic<unsigned> pendingGPUUploads_;

  public:
    Manager(int managedMemorySize, Runtime* runtime);

    std::future<void> loadModel(const std::string& name, const std::string& source);

    std::future<void> infer(const std::string& name, const std::string& inputName,
                DLTensor* input, int outIndex, DLTensor* output);

  private:

    std::future<void> loadToGPUAndInfer_(Model& model, const std::string& inputName,
                  DLTensor* input, int outIndex, DLTensor* output);

    std::future<void> infer_(Model& model, const std::string& inputName, DLTensor* input,
                  int outIndex, DLTensor* output);


  };

}  // namespace clockwork

#endif  // CLOCKWORK_MULTITENANT_MANAGER_H_
