#include "clockwork/model/cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "clockwork/tvm/pack_args.h"

#include <tvm/runtime/cuda_common.h>

namespace clockwork {
namespace cuda {


UnloadedCUDAModule::UnloadedCUDAModule(const char* &cuda_blob) {
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    uint64_t c = cuda_blob[i];
    nbytes |=  (c & 0xffUL) << (i * 8);
  }

  dmlc::MemoryFixedSizeStream fs(
      const_cast<char*>(cuda_blob + sizeof(nbytes)), static_cast<size_t>(nbytes));
  dmlc::Stream* stream = &fs;
  uint64_t size;
  CHECK(stream->Read(&size));
  CHECK(size == 1) << "Only expected one dev_mblob, found " << size;

  std::string tkey;
  CHECK(stream->Read(&tkey));
  std::string fkey = "module.loadbinary_" + tkey;
  CHECK(tkey == "cuda") << "Expected dev_mblob of type cuda, found " << tkey;

  stream->Read(&this->fmt);

  std::unordered_map<std::string, tvm::runtime::FunctionInfo> fmap;
  stream->Read(&fmap);

  this->functions.reserve(fmap.size());
  for (auto & e : fmap) {
    this->functions[e.first] = new UnloadedCUDAFunc(e.first, e.second);
  }

  std::string data;
  stream->Read(&this->data);
}

UnloadedCUDAModule::~UnloadedCUDAModule() {
  for (auto &e : this->functions) {
    delete(e.second);
  }
}

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
  CUresult result = cuModuleUnload(module);
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    const char *msg;
    cuGetErrorName(result, &msg);
    std::ostringstream os;
    os << "cuModuleUnload Error: " << msg << "\n";
    LOG(FATAL) << os.str();    
  }
}

void LoadedCUDAModule::unload() {
  for (auto &elem : this->source->functions) {
    elem.second->loaded = nullptr;
  }
  delete this;
}

tvm::runtime::PackedFunc* LoadedCUDAModule::getFunction(const std::string &name) {
  // This has been pushed to unloadedCudamodule
  LoadedCUDAFunc* f = functions[name];
  return &f->source->packed;
}

LoadedCUDAFunc::LoadedCUDAFunc(UnloadedCUDAFunc* source, CUfunction f) : source(source), f(f) {
  // packed = tvm::runtime::PackFuncVoidAddr(*this, source->info.arg_types);
}

void LoadedCUDAFunc::operator()(tvm::runtime::TVMArgs args,
                tvm::runtime::TVMRetValue* rv,
                void** void_args) const {
  CUstream strm = static_cast<CUstream>(tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream);
  tvm::runtime::ThreadWorkLoad wl = source->thread_axis_cfg_.Extract(args);
  // std::cout << "cuLaunchKernel " << wl.grid_dim(0) << " "
  //                                << wl.grid_dim(1) << " "
  //                                << wl.grid_dim(2) << " "
  //                                << wl.block_dim(0) << " "
  //                                << wl.block_dim(1) << " "
  //                                << wl.block_dim(2) << std::endl;
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
  // CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(strm)));
}

UnloadedCUDAFunc::UnloadedCUDAFunc(const std::string &name, const tvm::runtime::FunctionInfo &info) : name(name), info(info) {
  thread_axis_cfg_.Init(info.arg_types.size(), info.thread_axis_tags);
  packed = tvm::runtime::PackFuncVoidAddr(
    [this] (tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv, void** void_args) {
      CHECK(this->loaded != nullptr) << "Cannot call unloaded CUDA function";
      (*this->loaded)(args, rv, void_args);
    }, 
    info.arg_types
  );
}

LoadedCUDAFunc* UnloadedCUDAFunc::load(CUmodule &m) {
  CHECK(this->loaded == nullptr) << "Cannot load CUDA functions more than once";

  CUfunction f;

  CUresult result = cuModuleGetFunction(&f, m, name.c_str());
  if (result != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(result, &msg);
    LOG(FATAL)
        << "CUDAError: cuModuleGetFunction " << name
        << " failed with error: " << msg;
  }

  this->loaded = new LoadedCUDAFunc(this, f);
  return this->loaded;
}



}
}