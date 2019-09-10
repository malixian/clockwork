#ifndef _CLOCKWORK_SO_H_
#define _CLOCKWORK_SO_H_

#include <cstring>
#include <dlfcn.h>
#include "dmlc/logging.h"
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/packed_func.h>
#include <clockwork/tvm/meta_data.h>
#include <clockwork/tvm/thread_storage_scope.h>
#include <unistd.h>
#include <functional>
#include "clockwork/tvm/runtime_base.h"
#include "clockwork/cuda.h"

namespace clockwork {
namespace so {


class SharedObject {
public:
  const std::string name;
  void* lib_handle_{nullptr};

public:
  void* GetSymbol(const char* symbolName) {
    return dlsym(lib_handle_, symbolName);
  }

  SharedObject(const std::string &name) : name(name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LOCAL | RTLD_NOW);
    CHECK(lib_handle_ != nullptr) << "Failed to load SO " << name << dlerror();
  }

  ~SharedObject() {
    dlclose(lib_handle_);
  }

  template<typename T> void LinkFunctionPtr(void* funcPtr, T func) {
    if (funcPtr != nullptr) {
      *(reinterpret_cast<T*>(funcPtr)) = func;
    }
  }

  template<typename T> void LinkFunction(const char* funcNameInSo, T func) {
    LinkFunctionPtr(GetSymbol(funcNameInSo), func);
  }

};

class TVMWarmSharedObject;
class TVMHotSharedObject;

class TVMWarmSharedObject {
public:
  SharedObject so;
  clockwork::cuda::UnloadedCUDAModule* cuda;

  void* ptr_ModuleCtx;
  void* ptr_TVMBackendGetFuncFromEnv;
  void* ptr_TVMBackendAllocWorkspace;
  void* ptr_TVMBackendFreeWorkspace;

  TVMWarmSharedObject(const std::string &so_filename);
  ~TVMWarmSharedObject();

  TVMHotSharedObject* load();


  void linkHot(TVMHotSharedObject* hot);
  void linkErrors();

};

class TVMHotSharedObject {
public:
  clockwork::cuda::LoadedCUDAModule* cuda;
  TVMWarmSharedObject* warm;

  TVMHotSharedObject(TVMWarmSharedObject *warm);
  ~TVMHotSharedObject();

  void unload();
};


}
}
#endif