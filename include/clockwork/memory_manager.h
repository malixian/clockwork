#include "tvm/runtime/cuda_common.h"

class CUDAMemoryManager {
public:
	virtual void* alloc(int device, size_t nbytes, size_t alignment) = 0;
	virtual void free(int device, void* ptr) = 0;
};

class DefaultCUDAMemoryManager {
public:

	void* alloc(int device, size_t nbytes, size_t alignment) {
	    CUDA_CALL(cudaSetDevice(device));
	    CHECK_EQ(256 % alignment, 0U)
	        << "CUDA space is aligned at 256 bytes";
	    void *ret;
	    CUDA_CALL(cudaMalloc(&ret, nbytes));
	    return ret;

	}

	void free(int device, void* ptr) {
	    CUDA_CALL(cudaSetDevice(device));
	    CUDA_CALL(cudaFree(ptr));
	}

};

