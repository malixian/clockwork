
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
#include "clockwork/cuda.h"
#include "clockwork/so.h"

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
  uint64_t workspace_memory;
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


// Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args,
                                  int* type_codes,
                                  int num_args);


class WarmOp {
public:
  Op &op;

  std::vector<uint64_t> workspace_offsets;
  
  std::vector<uint64_t> offsets;
  std::vector<DLTensor*> input_tensors;
  std::vector<TVMValue> op_inputs;
  std::vector<int> op_tcodes;
  int size;

  BackendPackedCFunc f;

  WarmOp(Op &op, BackendPackedCFunc f) : op(op), f(f), workspace_offsets(op.workspace_allocs) {
    size = op.inputs.size();
    offsets.resize(size);
    op_inputs.resize(size);
    op_tcodes.resize(size);
    input_tensors.resize(size);

    for (unsigned i = 0; i < size; i++) {
      DLTensor* input = new DLTensor();
      input->data = nullptr;
      input->ctx = DLContext{kDLGPU, 0};
      input->ndim = op.inputs[i].shape.size();
      input->dtype = DLDataType{kDLFloat, 32, 1};
      input->shape = op.inputs[i].shape.data();
      input->strides = nullptr;
      input->byte_offset = 0;

      offsets[i] = op.inputs[i].offset;
      
      input_tensors[i] = input;
      op_inputs[i].v_handle = input;
      op_tcodes[i] = kArrayHandle;
    }
  }

  ~WarmOp() {
    for (unsigned i = 0; i < size; i++) {
      delete input_tensors[i];
    }
  }

  void call(void* baseptr) {
    for (unsigned i = 0; i < size; i++) {
      input_tensors[i]->data = static_cast<char*>(baseptr) + offsets[i];
    }

    // std::cout << size << " inputs" << std::endl;
    // for (unsigned i = 0; i < op_inputs.size(); i++) {
    //   DLTensor* tensor = static_cast<DLTensor*>(op_inputs[i].v_handle);
    //   std::cout << "Input " << i << " ndim=" << tensor->ndim << "shape=[";
    //   for (unsigned j = 0; j < tensor->ndim; j++) {
    //     std::cout << *(static_cast<int64_t*>(tensor->shape) + j) << " ";
    //   }
    //   std::cout << "]" << " datatype=" << tensor->dtype.code << "-" << tensor->dtype.bits << "-" << tensor->dtype.lanes << " stridesnull=" << (tensor->strides==nullptr) << " offset=" << tensor->byte_offset << std::endl;
    // }

    std::vector<void*> workspace_ptrs(workspace_offsets.size());
    for (unsigned i = 0; i < workspace_offsets.size(); i++) {
      workspace_ptrs[i] = static_cast<char*>(baseptr) + workspace_offsets[i];
    }

    clockwork::so::TVMBackendWorkspaceManager::Set(workspace_ptrs);
    int ret = (*f)(
      op_inputs.data(),
      op_tcodes.data(), 
      size
    );
    clockwork::so::TVMBackendWorkspaceManager::Clear();
    CHECK_EQ(ret, 0) << TVMGetLastError();
  }

};


class WarmModel {
public:
  int size;
  std::vector<WarmOp*> ops;

  WarmModel(MinModel &mm, clockwork::so::TVMWarmSharedObject* warm) : ops(mm.ops.size()), size(mm.total_memory) {
    // Extract the SO functions
    std::vector<BackendPackedCFunc> fs(mm.so_functions.size());
    for (unsigned i = 0; i < mm.so_functions.size(); i++) {
      fs[i] = reinterpret_cast<BackendPackedCFunc>(warm->so.GetSymbol(mm.so_functions[i].c_str()));
    }

    for (unsigned i = 0; i < mm.ops.size(); i++) {
      ops[i] = new WarmOp(mm.ops[i], fs[mm.ops[i].so_function]);

      // TODO: no need at the moment, but later, eager load cuda functions and workspace allocs
    }
  }

  ~WarmModel() {
    for (unsigned i = 0; i < ops.size(); i++) {
      delete ops[i];
    }
  }

  void call(void* baseptr) {
    for (unsigned i = 0; i < ops.size(); i++) {
      // std::cout << "Op " << i << " has ";
      ops[i]->call(baseptr);
    }
  }

};

}
}

#endif