
#ifndef TVM_CLOCKWORK_SERIALIZED_MODEL_H_
#define TVM_CLOCKWORK_SERIALIZED_MODEL_H_

#include <cstring>
#include <dlfcn.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "dmlc/logging.h"
#include <tvm/runtime/cuda_common.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/packed_func.h>
#include <clockwork/tvm/meta_data.h>
#include <clockwork/tvm/thread_storage_scope.h>
#include <unistd.h>
#include <functional>
#include "clockwork/tvm/runtime_base.h"

namespace clockwork {
namespace binary {

struct DLTensorDef {
	uint64_t offset;
	std::vector<int64_t> shape;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(offset),
        PODS_MDR(shape)
    )
};

struct Op {
	std::vector<DLTensorDef> inputs;
	unsigned so_function;
	std::vector<unsigned> cuda_functions;
	std::vector<uint64_t> workspace_allocs;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(inputs),
        PODS_MDR(so_function),
        PODS_MDR(cuda_functions),
        PODS_MDR(workspace_allocs)
    )
};

struct MinModel {
	uint64_t total_memory;
	uint64_t weights_memory;
	std::vector<std::string> so_functions;
	std::vector<std::string> cuda_functions;
	std::vector<Op> ops;

    PODS_SERIALIZABLE(1,         
        PODS_MDR(total_memory),
        PODS_MDR(weights_memory),
        PODS_MDR(so_functions),
        PODS_MDR(cuda_functions),
        PODS_MDR(ops)
    )
};

class SharedObject {
public:
  const std::string name;
  void* lib_handle_{nullptr};

public:
  SharedObject(const std::string name) : name(name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LOCAL | RTLD_NOW);
    CHECK(lib_handle_ != nullptr) << "Failed to load SO " << name << dlerror();
  }

  ~SharedObject() {
    dlclose(lib_handle_);
  }

  void* GetSymbol(const char* symbolName) {
    return dlsym(lib_handle_, symbolName);
  }

};

class UnloadedCUDAModule;
class UnloadedCUDAFunc;
class LoadedCUDAModule;
class LoadedCUDAFunc;


class UnloadedCUDAModule {
public:
  std::string fmt;
  std::string data;
  std::unordered_map<std::string, UnloadedCUDAFunc*> functions;
  UnloadedCUDAModule(SharedObject &so);
  LoadedCUDAModule* load();
};

class LoadedCUDAModule {
public:
  const UnloadedCUDAModule* source;
  CUmodule module;
  std::unordered_map<std::string, LoadedCUDAFunc*> functions;

  LoadedCUDAModule(const UnloadedCUDAModule* source, CUmodule module);
  ~LoadedCUDAModule();

  tvm::runtime::PackedFunc* getFunction(const std::string &name);

};

// a wrapped function class to get packed func.
class LoadedCUDAFunc {
public:
  UnloadedCUDAFunc* source;
  CUfunction f;
  tvm::runtime::PackedFunc packed;

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

  UnloadedCUDAFunc(const std::string &name, const tvm::runtime::FunctionInfo &info);

  LoadedCUDAFunc* load(CUmodule &m);
};


// Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args,
                                  int* type_codes,
                                  int num_args);




/** This is a pseudo-alternative to TVM's DSOmodule */
class TVMSharedObjectHandler {
private:
  SharedObject so;
  std::vector<std::string> fnames;

  std::vector<void*> fs;

public:
  UnloadedCUDAModule* unloadedCUDA;
  LoadedCUDAModule* loadedCUDA;

  static void __tvm_set_device(tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *ret) {
    DLDeviceType device_type = static_cast<DLDeviceType>(args[0].operator int());
    int device_id = args[1];
    std::cout << "__tvm_set_device " << device_id << std::endl;
  }

  static int TVMFuncCallP(TVMFunctionHandle func,
                TVMValue* args,
                int* arg_type_codes,
                int num_args,
                TVMValue* ret_val,
                int* ret_type_code) {
    std::cout << "TVMFuncCall " << std::endl;
    return TVMFuncCall(func, args, arg_type_codes, num_args, ret_val, ret_type_code);
  }

  static void TVMAPISetLastErrorP(const char* msg) {
    TVMAPISetLastError(msg);
  }

  static int TVMBackendGetFuncFromEnvP(void* mod_node, const char* func_name, TVMFunctionHandle *func) {
    API_BEGIN();
    if (strcmp(func_name, "__tvm_set_device") == 0) {
      tvm::runtime::PackedFunc* set_device = new tvm::runtime::PackedFunc(__tvm_set_device);
      *func = (TVMFunctionHandle)(set_device);
    } else {
      std::cout << "TVMBackendGetFuncFromEnv wheee " << func_name << std::endl;
      printf("%p\n", mod_node);
      TVMSharedObjectHandler* handler = static_cast<TVMSharedObjectHandler*>(mod_node);
      *func = (TVMFunctionHandle)(handler->loadedCUDA->getFunction(func_name));
      std::cout << "   done TVMBackendGetFuncFromEnv" << std::endl;
    }
    API_END();
  }

  static void* TVMBackendAllocWorkspaceP(int device_type,
                                 int device_id,
                                 uint64_t size,
                                 int dtype_code_hint,
                                 int dtype_bits_hint) {
    std::cout << "TVMBackendAllocWorkspaceP " << device_id << std::endl;
    CHECK(device_type == kDLGPU) << "TVM Backend alloc non-GPU workspace";

    // TODO: plug into managed memory
    CUDA_CALL(cudaSetDevice(device_id));

    void* ptr;
    CUDA_CALL(cudaMalloc(&ptr, size));
    return ptr;
  }


  static int TVMBackendFreeWorkspaceP(int device_type,
                              int device_id,
                              void* ptr) {
    std::cout << "TVMBackendFreeWorkspaceP" << std::endl;
    CHECK(device_type == kDLGPU) << "TVM Backend alloc non-GPU workspace";
    CUDA_CALL(cudaSetDevice(device_id));
    CUDA_CALL(cudaFree(ptr));
    return 0;
  }

  static int TVMBackendParallelLaunchP(FTVMParallelLambda flambda,
                                     void* cdata,
                                     int num_task) {
    CHECK(false) << "TVMBackendParallelLaunch unsupported";
  }

  static int TVMBackendParallelBarrierP(int task_id, TVMParallelGroupEnv* penv) {
    CHECK(false) << "TVMBackendParallelBarrier unsupported";
  }

  TVMSharedObjectHandler(const std::string name, std::vector<std::string> toLoad) : so(name), fs(toLoad.size()) {
    // Eagerly extract all of the op functions
    for (unsigned i = 0; i < toLoad.size(); i++) {
      const char* name = toLoad[i].c_str();
      fs[i] = so.GetSymbol(name);
    }

    // Eagerly extract the CUDA module code (but don't load it to device yet)
    unloadedCUDA = new UnloadedCUDAModule(so);

    // Insert pointer to this object for module handler
    LinkFunction(tvm::runtime::symbol::tvm_module_ctx, this);

    // Insert pointers into the SO for callbacks
    LinkFunction("__TVMFuncCall", TVMFuncCallP);
    LinkFunction("__TVMAPISetLastError", TVMAPISetLastErrorP);
    LinkFunction("__TVMBackendGetFuncFromEnv", TVMBackendGetFuncFromEnvP);
    LinkFunction("__TVMBackendAllocWorkspace", TVMBackendAllocWorkspaceP);
    LinkFunction("__TVMBackendFreeWorkspace", TVMBackendFreeWorkspaceP);
    LinkFunction("__TVMBackendParallelLaunch", TVMBackendParallelLaunchP);
    LinkFunction("__TVMBackendParallelBarrier", TVMBackendParallelBarrierP);
  }

  template<typename T> void LinkFunction(const char* funcNameInSo, T func) {
    if (T* fp = reinterpret_cast<T*>(so.GetSymbol(funcNameInSo))) {
      *fp = func;
    }
  }

  void* &operator[](int i) {
    CHECK(i >= 0 && i < fs.size()) << "Function lookup index out of bounds";
    return fs[i];
  }

};


class Test {
public:

static void testModel(MinModel &model) {
  void* ptr;
  CUDA_CALL(cudaMalloc(&ptr, model.total_memory));

  TVMSharedObjectHandler so("/home/jcmace/modelzoo/resnet50/tesla-m40_batchsize1/tvm-model.so", model.so_functions);

  so.loadedCUDA = so.unloadedCUDA->load();

  for (unsigned i = 0; i < model.ops.size(); i++) {
    Op op = model.ops[i];
    std::cout << "op " << i << " " << model.so_functions[op.so_function] << std::endl;

    bool print_op_input_info = false;
    if (print_op_input_info) {
      std::cout << "Op " << i << " is " << model.so_functions[op.so_function] << " with " << op.inputs.size() << " inputs" << std::endl;
      for (unsigned j = 0; j < op.inputs.size(); j++) {
        std::cout << "   Input " << j << " at offset " << op.inputs[j].offset << " shape ";
        for (unsigned k = 0; k < op.inputs[j].shape.size(); k++) {
          std::cout << op.inputs[j].shape[k] << " ";
        }
        std::cout << std::endl;
      }
    }

    std::vector<TVMValue> values(op.inputs.size());
    std::vector<int> tcodes(op.inputs.size());
    for (unsigned j = 0; j < op.inputs.size(); j++) {
      DLTensor* input = new DLTensor();
      input->data = ptr + op.inputs[j].offset;
      input->ctx = DLContext{kDLGPU, 0};
      //input->ctx = DLContext{kDLGPU, i};
      input->ndim = op.inputs[j].shape.size();
      input->dtype = DLDataType{kDLFloat, 32, 1};
      input->shape = op.inputs[j].shape.data();
      input->strides = nullptr;
      input->byte_offset = 0;

      values[j].v_handle = input;
      tcodes[j] = kArrayHandle;
    }

    tvm::runtime::TVMRetValue rv;
    tvm::runtime::TVMArgs targs(values.data(), tcodes.data(), static_cast<int>(values.size()));

    void* f = so[op.so_function];


    int ret = (*reinterpret_cast<BackendPackedCFunc>(f))(
      const_cast<TVMValue*>(values.data()), 
      const_cast<int*>(tcodes.data()), 
      static_cast<int>(values.size())
    );
    CHECK_EQ(ret, 0) << TVMGetLastError();

  }
}
};



}
}

#endif