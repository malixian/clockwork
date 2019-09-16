#ifndef _CLOCKWORK_CUDA_H_
#define _CLOCKWORK_CUDA_H_

#include <unordered_map>
#include <string>
#include <cuda.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include "clockwork/tvm/runtime_base.h"
#include "clockwork/tvm/meta_data.h"
#include "clockwork/tvm/thread_storage_scope.h"

namespace clockwork {
namespace cuda {

class UnloadedCUDAModule;
class UnloadedCUDAFunc;
class LoadedCUDAModule;
class LoadedCUDAFunc;


class UnloadedCUDAModule {
public:
  std::string fmt;
  std::string data;
  std::unordered_map<std::string, UnloadedCUDAFunc*> functions;
  UnloadedCUDAModule(const char* &cuda_blob);
  ~UnloadedCUDAModule();
  LoadedCUDAModule* load();
};

class LoadedCUDAModule {
public:
  const UnloadedCUDAModule* source;
  CUmodule module;
  std::unordered_map<std::string, LoadedCUDAFunc*> functions;

  LoadedCUDAModule(const UnloadedCUDAModule* source, CUmodule module);
  ~LoadedCUDAModule();

  void unload();

  tvm::runtime::PackedFunc* getFunction(const std::string &name);

};

// a wrapped function class to get packed func.
class LoadedCUDAFunc {
public:
  UnloadedCUDAFunc* source;
  CUfunction f;

  LoadedCUDAFunc(UnloadedCUDAFunc* source, CUfunction f);

  void operator()(tvm::runtime::TVMArgs args,
                  tvm::runtime::TVMRetValue* rv,
                  void** void_args) const;
};

class UnloadedCUDAFunc {
public:
  const std::string name;
  const tvm::runtime::FunctionInfo info;
  tvm::runtime::ThreadAxisConfig thread_axis_cfg_;
  LoadedCUDAFunc* loaded = nullptr;
  tvm::runtime::PackedFunc packed;

  UnloadedCUDAFunc(const std::string &name, const tvm::runtime::FunctionInfo &info);

  LoadedCUDAFunc* load(CUmodule &m);
};

}
}
#endif