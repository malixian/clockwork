#include "clockwork/util/tvm_util.h"
#include "tvm/runtime/cuda_common.h"

namespace clockwork {
namespace tvmutil {	

void initializeTVMCudaStream() {
    CUDA_CALL(cudaSetDevice(0));
	cudaStream_t stream;	
	CUDA_CALL(cudaStreamCreate(&stream));
	tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream = stream;
	tvm::runtime::CUDAThreadEntry::ThreadLocal()->stream = stream;
}

}
}