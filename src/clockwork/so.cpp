#include "clockwork/so.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/runtime/cuda_common.h>

namespace clockwork {
namespace so {

int TVMFuncCallProxy(TVMFunctionHandle func,
                 TVMValue* args,
                 int* arg_type_codes,
                 int num_args,
                 TVMValue* ret_val,
                 int* ret_type_code) {
    std::cout << "TVMFuncCall " << std::endl;
    return TVMFuncCall(func, args, arg_type_codes, num_args, ret_val, ret_type_code);
}

void TVMAPISetLastErrorProxy(const char* msg) {
	TVMAPISetLastError(msg); // Just call the TVM api for
}

void __tvm_set_device(tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *ret) {
    DLDeviceType device_type = static_cast<DLDeviceType>(args[0].operator int());
    int device_id = args[1];
    std::cout << "__tvm_set_device " << device_id << std::endl;
}
tvm::runtime::PackedFunc* set_device = new tvm::runtime::PackedFunc(__tvm_set_device);

int TVMBackendGetFuncFromEnvHot(void* mod_node, const char* func_name, TVMFunctionHandle *func) {
	API_BEGIN();
	if (strcmp(func_name, "__tvm_set_device") == 0) {
	  *func = (TVMFunctionHandle)(set_device);
	} else {
	  std::cout << "TVMBackendGetFuncFromEnv wheee " << func_name << std::endl;
	  printf("%p\n", mod_node);
	  TVMHotSharedObject* hot = static_cast<TVMHotSharedObject*>(mod_node);
	  *func = (TVMFunctionHandle)(hot->cuda->getFunction(func_name));
	  std::cout << "   done TVMBackendGetFuncFromEnv" << std::endl;
	}
	API_END();
}

void* TVMBackendAllocWorkspaceHot(int device_type,
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


int TVMBackendFreeWorkspaceHot(int device_type,
                             int device_id,
                             void* ptr) {
	std::cout << "TVMBackendFreeWorkspaceP" << std::endl;
	CHECK(device_type == kDLGPU) << "TVM Backend alloc non-GPU workspace";
	CUDA_CALL(cudaSetDevice(device_id));
	CUDA_CALL(cudaFree(ptr));
	return 0;
}

int TVMBackendGetFuncFromEnvError(void* mod_node, const char* func_name, TVMFunctionHandle *func) {
	API_BEGIN();
	CHECK(false) << "TVMBackendGetFuncFromEnv invoked on warm model";
	API_END();
}

void* TVMBackendAllocWorkspaceError(int device_type,
                                int device_id,
                                uint64_t size,
                                int dtype_code_hint,
                                int dtype_bits_hint) {
	CHECK(false) << "TVMBackendAllocWorkspace invoked on warm model";
	return nullptr;
}


int TVMBackendFreeWorkspaceError(int device_type,
                             int device_id,
                             void* ptr) {
	CHECK(false) << "TVMBackendFreeWorkspace invoked on warm model";
	return 0;
}

int TVMBackendParallelLaunchError(FTVMParallelLambda flambda,
	                          void* cdata,
	                          int num_task) {
	CHECK(false) << "TVMBackendParallelLaunch unsupported";
}

int TVMBackendParallelBarrierError(int task_id, TVMParallelGroupEnv* penv) {
	CHECK(false) << "TVMBackendParallelBarrier unsupported";
}

TVMWarmSharedObject::TVMWarmSharedObject(const std::string &so_filename, std::vector<std::string> &toLoad) : so(so_filename), fs(toLoad.size()) {
	// Eagerly extract all of the op functions
    for (unsigned i = 0; i < toLoad.size(); i++) {
      const char* name = toLoad[i].c_str();
      fs[i] = so.GetSymbol(name);
    }

    // Extract the CUDA module blob
    const char* cuda_blob = reinterpret_cast<const char*>(so.GetSymbol(tvm::runtime::symbol::tvm_dev_mblob));
    CHECK(cuda_blob != nullptr) << "Could not find " << tvm::runtime::symbol::tvm_dev_mblob 
                                << " in SO " << so_filename;
    this->cuda = new clockwork::cuda::UnloadedCUDAModule(cuda_blob);

    // Extract the function pointers for functions that get swapped in and out
    ptr_ModuleCtx = so.GetSymbol(tvm::runtime::symbol::tvm_module_ctx);
    ptr_TVMBackendGetFuncFromEnv = so.GetSymbol("__TVMBackendGetFuncFromEnv");
    ptr_TVMBackendAllocWorkspace = so.GetSymbol("__TVMBackendAllocWorkspace");
    ptr_TVMBackendFreeWorkspace = so.GetSymbol("__TVMBackendFreeWorkspace");

    // Insert function pointers for functions that DONT get swapped in and out
    so.LinkFunction("__TVMFuncCall", TVMFuncCallProxy);
    so.LinkFunction("__TVMAPISetLastError", TVMAPISetLastErrorProxy);
    so.LinkFunction("__TVMBackendParallelLaunch", TVMBackendParallelLaunchError);
    so.LinkFunction("__TVMBackendParallelBarrier", TVMBackendParallelBarrierError);

    // Insert error functions for functions that shouldn't be called until hot
    this->linkErrors();
}

void TVMWarmSharedObject::linkHot(TVMHotSharedObject* hot) {
    // Insert pointer to the hot SO for module context
    so.LinkFunctionPtr(ptr_ModuleCtx, hot);

    // Insert hot functions
    so.LinkFunctionPtr(ptr_TVMBackendGetFuncFromEnv, TVMBackendGetFuncFromEnvHot);
    so.LinkFunctionPtr(ptr_TVMBackendAllocWorkspace, TVMBackendAllocWorkspaceHot);
    so.LinkFunctionPtr(ptr_TVMBackendFreeWorkspace, TVMBackendFreeWorkspaceHot);
}

void TVMWarmSharedObject::linkErrors() {
    // Remove module ctx
    so.LinkFunctionPtr(ptr_ModuleCtx, (TVMHotSharedObject*)nullptr);

    // Insert error functions for functions that shouldn't be called until hot
    so.LinkFunctionPtr(ptr_TVMBackendGetFuncFromEnv, TVMBackendGetFuncFromEnvError);
    so.LinkFunctionPtr(ptr_TVMBackendAllocWorkspace, TVMBackendAllocWorkspaceError);
    so.LinkFunctionPtr(ptr_TVMBackendFreeWorkspace, TVMBackendFreeWorkspaceError);
}

TVMHotSharedObject* TVMWarmSharedObject::load() {
	return new TVMHotSharedObject(this);
}

TVMHotSharedObject::TVMHotSharedObject(TVMWarmSharedObject *warm) : warm(warm) {
	// Link hot code to this
	warm->linkHot(this);

    // Load CUDA code onto device
	this->cuda = warm->cuda->load();
}

TVMHotSharedObject::~TVMHotSharedObject() {
    // Unlink hot code
    warm->linkErrors();

    // Unload CUDA code from device
	this->cuda->unload();
}

void TVMHotSharedObject::unload() {
	delete this;
}


}
}