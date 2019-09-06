/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef CLOCKWORK_MULTITENANT_MODEL_MANAGER_H_
#define CLOCKWORK_MULTITENANT_MODEL_MANAGER_H_

#include <tvm/runtime/managed_cuda_device_api.h>
#include <climits>
#include <cstdlib>
#include <map>
#include <unistd.h>

namespace clockwork {

  static int kEvictionRate = 0;

  enum ModelStatus {
    READY = 0,
    IN_USE,
    EVICTED
  };

  struct Model {
    tvm::runtime::Module mod;
    std::string name;
    ModelStatus status;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_use;

    std::map<std::string, tvm::runtime::PackedFunc> func_cache_;

    tvm::runtime::PackedFunc& GetFunction(const std::string& name) {
      if (!func_cache_.count(name)) {
        func_cache_[name] = mod.GetFunction(name);
      }
      return func_cache_[name];
    }

    // need this for compilation, but it should never actually be called
    Model () { CHECK(false) << "Model instantiated without actual code"; }

    Model(tvm::runtime::Module&& module, const std::string& n) : mod(module),
                                                   name(n),
                                                   status(ModelStatus::READY),
                                                   last_use(std::chrono::high_resolution_clock::now()) {
                                                     // warm up function cache on creation
                                                     GetFunction("load_params_contig");
                                                     GetFunction("load_to_device");
                                                     GetFunction("get_contig_context");
                                                     GetFunction("set_input");
                                                     GetFunction("run");
                                                     GetFunction("get_output");
                                                     GetFunction("evicted");
                                                   }
  };

  class ModelManager : public tvm::runtime::EvictionHandler {
  public:
    ModelManager() {
      tvm::runtime::ManagedCUDADeviceAPI::Global()->SetEvictionHandler(this);
    }

    void insertFauxEviction(const std::string& name) {
      this->faux_evictions.insert(name);
    }

    typedef std::pair<std::list<tvm::runtime::MemBlock>::const_iterator, std::list<tvm::runtime::MemBlock>::const_iterator> it_pair;
    it_pair evict(std::list<tvm::runtime::MemBlock>& mem, size_t nbytes) final {
      // find the shortest range with the memory we need and longest average
      // time since it was used
      auto evict_time = std::chrono::high_resolution_clock::now();
      it_pair ret = std::make_pair(mem.end(), mem.end());
      int avg_time = INT_MIN;

      // for each block, find the smallest subset of blocks covering it and
      // giving enough memory
      auto first = mem.begin();
      for (; first != mem.end(); first++) {
        size_t total = first->size;
        size_t count = 0;
        int total_time = 0;

        if (!first->isfree) {
          count++;
          mapLock_.lock();
          total_time = std::chrono::duration_cast<std::chrono::milliseconds>(evict_time - models_[first->owner].last_use).count();
          mapLock_.unlock();
        }

        auto second = std::next(first);
        for (; second != mem.end(); second++) {
          if (total < nbytes) {
            total += second->size;
            if (!second->isfree) {
              count++;
              mapLock_.lock();
              total_time = std::chrono::duration_cast<std::chrono::milliseconds>(evict_time - models_[second->owner].last_use).count();
              mapLock_.unlock();
            }
          } else {
            break;
          }
        }

        if (total >= nbytes) {
          int new_avg = total_time / count;
          if (new_avg > avg_time) {
            avg_time = new_avg;
            ret = std::make_pair(first, second);
          }
        } else {
          break; // if we couldn't find a set of blocks large enough, then we
                 // won't ever and we can let the caller handle this
        }
      }

      // wait for models to finish running if in use then mark them as evicted
      int count = 0;
      for (auto it = ret.first; it != ret.second; it++) {
        count++;
        if (!it->isfree) {
          mapLock_.lock();
          std::mutex& mlock = modelLocks_[it->owner];
          Model& model = models_[it->owner];
          mapLock_.unlock();

          mlock.lock();
          model.status = ModelStatus::EVICTED;
          model.GetFunction("evicted")();
          mlock.unlock();
        }
      }

      return ret;
    }

    void lockModel(const std::string& name);

    void loadModelFromDisk(const std::string& name, const std::string& source);

    void instantiateModel(const std::string& name);

    Model& getModel(const std::string& name);

    void releaseModel(const std::string& name);

  private:

    // map from name to model and lock
    std::map<std::string, Model> models_;
    std::map<std::string, std::mutex> modelLocks_;

    // lock for both models_ and modelLocks_ since we have reads and writes to
    // both on multiple threads
    std::mutex mapLock_;

    // temp storage for loading from disk
    std::mutex sourceLock_;
    std::map<std::string, std::tuple<std::string*, std::string*, tvm::runtime::Module>> modelSource_;
  };

}  // namespace clockwork

#endif  // CLOCKWORK_MULTITENANT_MODEL_MANAGER_H_
