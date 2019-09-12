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
    if (this->modelLocks_.count(name) == 0) {
      this->modelLocks_.emplace(name, new std::mutex());
    }
    this->modelLocks_.at(name)->lock();
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

    modelSource_.emplace(name, std::make_tuple(
        json_data, params_data, std::move(mod_syslib)
    ));
  }

  void ModelManager::instantiateModel(const std::string& name) {
    tbb::concurrent_hash_map<std::string, std::tuple<std::string*, std::string*, tvm::runtime::Module>>::accessor acc;
    CHECK(modelSource_.find(acc, name)) << "Tried to instantiate a model whose source is not currently loaded.";
    std::string* json_data = std::get<0>((*acc).second);
    std::string* params_data = std::get<1>((*acc).second);
    tvm::runtime::Module& mod_syslib = std::get<2>((*acc).second);

    TVMByteArray params_arr;
    params_arr.data = params_data->c_str();
    params_arr.size = params_data->length();

    // create the model and set its status as evicted

    std::shared_ptr<tvm::runtime::DecoupledGraphRuntime> rt = DecoupledGraphRuntimeCreateDirect(*json_data, mod_syslib, device_type, device_id);

    this->models_.emplace(std::piecewise_construct,
                          std::forward_as_tuple(name),
                          std::forward_as_tuple(tvm::runtime::Module(rt), name));
    tvm::runtime::PackedFunc load_params = this->models_[name].GetFunction("load_params_contig");
    this->models_[name].status = ModelStatus::EVICTED;

    load_params(params_arr);

    this->modelLocks_.at(name)->unlock();

    delete json_data;
    delete params_data;

    modelSource_.erase(acc);
  }

  Model& ModelManager::getModel(const std::string& name) {
    CHECK(models_.count(name) == 1) << name << " does not name a model previously loaded.";

    this->modelLocks_.at(name)->lock();

    Model& model = models_[name];

    return model;
  }

  void ModelManager::releaseModel(const std::string& name) {
    CHECK(models_.count(name) == 1) << name << " does not name a model previously loaded.";

    this->modelLocks_.at(name)->unlock();
  }

}  // namespace clockwork
