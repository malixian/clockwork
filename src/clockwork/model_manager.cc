#include <clockwork/model_manager.h>

#include <dlpack/dlpack.h>
#include <chrono>
#include <future>
#include "clockwork/tvm/decoupled_graph_runtime.h"

#include "util/shmem.h"

namespace clockwork {

  const int device_type = kDLGPU;
  const int device_id = 0;

  void ModelManager::lockModel(const std::string& name) {
    this->mapLock_.lock();
    this->modelLocks_[name].lock();
    this->mapLock_.unlock();
  }

  void ModelManager::loadModelFromDisk(const std::string& name, const std::string& source) {
    const tvm::runtime::PackedFunc load_module(*tvm::runtime::Registry::Get("module.loadfile_so"));

    // load graph structure
    std::ifstream json_in(source + ".json", std::ios::in);  // read as text
    std::string* json_data = new std::string((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // load params contiguous block
    std::ifstream params_in(source + ".params_contig", std::ios::binary);
    std::string* params_data = new std::string((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // module containing host and dev code
    tvm::runtime::Module mod_syslib = load_module(copy_to_memory(source + ".so"), "so");

    sourceLock_.lock();
    modelSource_.emplace(name, std::make_tuple(
        json_data, params_data, std::move(mod_syslib)
    ));
    sourceLock_.unlock();
  }

  void ModelManager::instantiateModel(const std::string& name) {
    sourceLock_.lock();
    std::string* json_data = std::get<0>(modelSource_[name]);
    std::string* params_data = std::get<1>(modelSource_[name]);
    tvm::runtime::Module& mod_syslib = std::get<2>(modelSource_[name]);
    sourceLock_.unlock();

    TVMByteArray params_arr;
    params_arr.data = params_data->c_str();
    params_arr.size = params_data->length();

    // create the model and set its status as in use
    this->mapLock_.lock();

    std::shared_ptr<tvm::runtime::DecoupledGraphRuntime> rt = DecoupledGraphRuntimeCreateDirect(*json_data, mod_syslib, device_type, device_id);
    
    this->models_.emplace(std::piecewise_construct,
                          std::forward_as_tuple(name),
                          std::forward_as_tuple(tvm::runtime::Module(rt), name));
    tvm::runtime::PackedFunc load_params = this->models_[name].GetFunction("load_params_contig");
    this->models_[name].status = ModelStatus::EVICTED;
    this->mapLock_.unlock();

    load_params(params_arr);

    this->mapLock_.lock();
    this->modelLocks_[name].unlock();
    this->mapLock_.unlock();

    delete json_data;
    delete params_data;

    sourceLock_.lock();
    modelSource_.erase(name);
    sourceLock_.unlock();
  }

  Model& ModelManager::getModel(const std::string& name) {
    mapLock_.lock();

    CHECK(models_.count(name) == 1) << name << " does not name a model previously loaded.";

    while (!modelLocks_[name].try_lock()) {
      mapLock_.unlock();
      mapLock_.lock();
    }

    Model& model = models_[name];

    mapLock_.unlock();

    return model;
  }

  void ModelManager::releaseModel(const std::string& name) {
    mapLock_.lock();

    CHECK(models_.count(name) == 1) << name << " does not name a model previously loaded.";

    modelLocks_[name].unlock();

    mapLock_.unlock();
  }

}  // namespace clockwork
