/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief components for pipelining loading and inferencing
 * \file executor.h
 */
#ifndef CLOCKWORK_MULTITENANT_MODEL_MANAGER_H_
#define CLOCKWORK_MULTITENANT_MODEL_MANAGER_H_

#include <tvm/runtime/managed_cuda_device_api.h>
#include <clockwork/clockwork.h>
#include <climits>
#include <cstdlib>
#include <map>
#include <unistd.h>

namespace clockwork {

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

  struct MemBlock {
    bool isfree = false;
    size_t size = 0;
    void* start = nullptr;
    std::string owner = "";
    MemBlock(bool isfree, size_t size, void* start) : isfree(isfree), size(size), start(start) {}
  };

  class EvictionHandler {
  public:
    virtual std::pair<std::list<MemBlock>::const_iterator,
                      std::list<MemBlock>::const_iterator>
                      evict(std::list<MemBlock>& mem, size_t nbytes) = 0;

    std::set<std::string> faux_evictions; // for setting eviction rates
  };

  class ClockworkMemoryManager : public tvm::runtime::CUDAMemoryManager {
public:
  const size_t reservationSize;
    EvictionHandler* ev_handler_ = nullptr;
    std::map<int,std::mutex> mem_locks_;
    std::map<int,std::list<MemBlock>> memory_;

    ClockworkMemoryManager(size_t reservationSize, EvictionHandler* ev_handler) : ev_handler_(ev_handler), reservationSize(reservationSize) {}

    ~ClockworkMemoryManager() {
      // if we don't wait on everything to be coalesced we may have someone trying
      // to free this pooled memory after the api has been destructed
      for (auto& it : memory_) {
        auto& mem = it.second;
        while (mem.size() > 1) {}
        std::lock_guard<std::mutex> lock(mem_locks_[it.first]);
        CUDA_CALL(cudaSetDevice(it.first));
        CUDA_CALL(cudaFree(it.second.begin()->start));
      }     
    }

  /* brief: mark the described block as owned by @param name */
  void ClaimOwnership(int device, const void* address, const std::string& name) {
    std::list<MemBlock>& mem = memory_[device];

    auto it = mem.begin();
    for (; it != mem.end(); it++) {
      if (it->start == address) {
        it->owner = name;
        break;
      }
    }

    CHECK (it != mem.end()) << "Block beginning at " << address << " does not "
                            << "exist on device " << device << std::endl;
    mem_locks_[device].unlock();
  }

  void* alloc(int device, size_t nbytes, size_t alignment) {
      std::list<MemBlock>& mem = memory_[device];
      mem_locks_[device].lock(); // don't unlock until memory is claimed
      void *ret;

      // in case memory was evicted because of our preset eviction rate, mark it
      // as free right now
      for (const auto& name : ev_handler_->faux_evictions) {
        for (auto& block : mem) {
          if (block.owner == name) {
            block.isfree = true;
            break;
          }
        }
      }

      ev_handler_->faux_evictions.clear();

      if (mem.size() == 0) { // initial reservation needs to be made
        CUDA_CALL(cudaMalloc(&ret, reservationSize));
        MemBlock b(true, reservationSize, ret);
        mem.push_back(b);
      }

      // find a block of free memory with at least nbytes
      auto it = mem.begin();
      for (; it != mem.end(); it++) {
        if (it->isfree && (it->size >= nbytes)) {
          break;
        }
      }

      if (it == mem.end()) {
        // we need to evict something (assume we won't ever just expand to simplify management)
        if (ev_handler_ != nullptr) {
          auto range = ev_handler_->evict(mem, nbytes);
          auto start = range.first;
          auto end = range.second;
          ret = range.first->start;
          size_t total = 0;
          int count = 0;
          for (; range.first != range.second; range.first++) {
            count++;
            total += range.first->size;
          }

          CHECK(total >= nbytes) << "Eviction Handler did not free enough memory. "
                                 << "Needed: " << nbytes << ", got: " << total;

          MemBlock allocated(false, nbytes, ret);
          MemBlock stillfree(true, total - nbytes, ret + nbytes);
          mem.insert(start, allocated);
          if (stillfree.size > 0) {
            mem.insert(start, stillfree);
          }
          auto it = std::prev(start); // pointing to leftover free block if it's there
          mem.erase(start, end);

          if (stillfree.size > 0) {
            coalesceMemBlocks(it, mem);
          }
        }
      } else {
        ret = it->start;
        MemBlock allocated(false, nbytes, ret);
        MemBlock stillfree(true, it->size - nbytes, ret + nbytes);
        mem.insert(it, allocated);
        if (stillfree.size > 0) {
          mem.insert(it, stillfree);
        }
        mem.erase(it);
      }

      return ret;
  }

    void SetEvictionHandler(EvictionHandler* eh) {
      ev_handler_ = eh;
    }

  void free(int device, void* ptr) {
      std::list<MemBlock>& mem = memory_[device];
      std::lock_guard<std::mutex> lock(mem_locks_[device]);
      auto it = mem.begin();

      // find block and mark it is as free
      for (; it != mem.end(); it++) {
        if (it->start == ptr) {
          it->isfree = true;
          it->owner = "";
          break;
        }
      }

      coalesceMemBlocks(it, mem);
  }

  void printMemList(const std::list<MemBlock>& mem) {
  for (auto it = mem.begin(); it != mem.end(); it++) {
    std::cout << "<" << it->start << "|" << it->isfree << "|" << it->size << ">" << "->";
  }
  std::cout << "END\n";
  }

  void coalesceMemBlocks(std::list<MemBlock>::const_iterator it, std::list<MemBlock>& mem) {
  // coalesce free memory around this block into a single block
  auto start = it;
  for (; start != mem.begin(); start--) {
    if (!start->isfree) {
      break;
    }
  }

  // incase it hit the front of the list and it isn't free
  if (!start->isfree) start++;

  auto end = it;
  for (; end != mem.end(); end++) {
    if (!end->isfree) {
      break;
    }
  }

  // if there is an adjacent range of blocks that are all free, mush them into
  // a single block
  if (start != end) {
    size_t total = 0;
    for (auto it = start; it != end; it++) {
      total += it->size;
    }
    MemBlock newfree(true, total, start->start);
    mem.insert(start, newfree);
    mem.erase(start, end);
  }
  }
};

  class ModelManager : public EvictionHandler {
  public:
    void insertFauxEviction(const std::string& name) {
      this->faux_evictions.insert(name);
    }

    typedef std::pair<std::list<MemBlock>::const_iterator, std::list<MemBlock>::const_iterator> it_pair;
    it_pair evict(std::list<MemBlock>& mem, size_t nbytes) final {
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
