#include "clockwork/util/tvm_util.h"
#include "tvm/runtime/cuda_common.h"

namespace clockwork {
namespace tvmutil {	

void initializeTVMCudaStream() {
	cudaStream_t stream;
	CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream = stream;
	tvm::runtime::CUDAThreadEntry::ThreadLocal()->stream = stream;
}

}
}