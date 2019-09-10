
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
  std::vector<uint64_t> offsets;
  std::vector<DLTensor*> input_tensors;

  std::vector<TVMValue> op_inputs;
  std::vector<int> op_tcodes;
  int size;

  BackendPackedCFunc f;

  WarmOp(Op &op, BackendPackedCFunc f) : op(op), f(f) {
    size = op.inputs.size();
    offsets.reserve(size);
    op_inputs.reserve(size);
    op_tcodes.reserve(size);
    input_tensors.reserve(size);

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

  void call(void* baseptr) {
    for (unsigned i = 0; i < size; i++) {
      input_tensors[i]->data = baseptr + offsets[i];
    }

    std::cout << "invoke function " << op.so_function << std::endl;

    int ret = (*f)(
      const_cast<TVMValue*>(op_inputs.data()),
      const_cast<int*>(op_tcodes.data()), 
      size
    );
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

  void call(void* baseptr) {
    for (unsigned i = 0; i < ops.size(); i++) {
      std::cout << "Call op " << i << std::endl;
      ops[i]->call(baseptr);
    }
  }

};

}
}

#endif