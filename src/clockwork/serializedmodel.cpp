

#include "clockwork/serializedmodel.h"

namespace clockwork {
namespace binary {

LoadedCUDAModule* UnloadedCUDAModule::load() {
  CUmodule module;
  CUresult result = cuModuleLoadData(&module, data.c_str());
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    const char *msg;
    cuGetErrorName(result, &msg);
    std::ostringstream os;
    os << "cuModuleLoadData Error: " << msg << "\n";
    LOG(FATAL) << os.str();    
  }
  return new LoadedCUDAModule(this, module);  
}

LoadedCUDAModule::LoadedCUDAModule(
      const UnloadedCUDAModule* source, 
      CUmodule module
    ) : source(source), module(module) {
  functions.reserve(source->functions.size());

  for (auto &e : source->functions) {
    functions[e.first] = e.second->load(module);
  }
}

LoadedCUDAModule::~LoadedCUDAModule() {
  for (auto &e : functions) {
    delete e.second;
  }
}

LoadedCUDAFunc::LoadedCUDAFunc(UnloadedCUDAFunc* source, CUfunction f) : source(source), f(f) {}

void LoadedCUDAFunc::operator()(tvm::runtime::TVMArgs args,
                tvm::runtime::TVMRetValue* rv,
                void** void_args) const {
  CUstream strm = static_cast<CUstream>(tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream);
  tvm::runtime::ThreadWorkLoad wl = source->thread_axis_cfg_.Extract(args);
  CUDA_LOG(
  CUresult result = cuLaunchKernel(
      f,
      wl.grid_dim(0),
      wl.grid_dim(1),
      wl.grid_dim(2),
      wl.block_dim(0),
      wl.block_dim(1),
      wl.block_dim(2),
      0, strm, void_args, 0);
  )
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    const char *msg;
    cuGetErrorName(result, &msg);
    std::ostringstream os;
    os << "cuLaunchKernel Error: " << msg << "\n"
       << " grid=(" << wl.grid_dim(0) << ","
       << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
       << " block=(" << wl.block_dim(0) << ","
       << wl.block_dim(1) << "," << wl.block_dim(2) << ")\n";
    os << "// func_name=" << source->info.name << "\n";
    LOG(FATAL) << os.str();
  }
}

UnloadedCUDAFunc::UnloadedCUDAFunc(const tvm::runtime::FunctionInfo &info) : info(info) {
    thread_axis_cfg_.Init(info.arg_types.size(), info.thread_axis_tags);
}

LoadedCUDAFunc* UnloadedCUDAFunc::load(CUmodule &m) {
  CUfunction f;

  CUresult result = cuModuleGetFunction(&f, m, info.name.c_str());
  if (result != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(result, &msg);
    LOG(FATAL)
        << "CUDAError: cuModuleGetFunction " << info.name
        << " failed with error: " << msg;
  }

  return new LoadedCUDAFunc(this, f);
}



}
}