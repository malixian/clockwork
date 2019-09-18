#ifndef _CLOCKWORK_MODEL_IMPL_2_H_
#define _CLOCKWORK_MODEL_IMPL_2_H_

#include <string>
#include <array>
// #include <tvm/runtime/ndarray.h>
// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/registry.h>
// #include <tvm/runtime/serializer.h>
#include "clockwork/model.h"
#include "clockwork/modeldef.h"
#include "clockwork/model/memfile.h"
#include "clockwork/model/so.h"
#include <cuda_runtime.h>
#include "clockwork/util.h"

#define MAX_OUTSTANDING_EVENTS 2

namespace clockwork{
namespace model {

// TVM Function signature for generated packed function in shared library
typedef int (*OpFunc)(void* args, int* type_codes, int num_args);

struct OpExec {
	PageMappedOpDef* spec;

	unsigned num_inputs;
	std::vector<DLTensor> input_tensors;
	std::vector<TVMValue> func_inputs;
	std::vector<int> func_tcodes;

	std::vector<void*> workspace_ptrs;

	std::string so_function_name;
	OpFunc f;
};

class Model {
private:


private:

	/* These events are used to rate-limit submission of asynchronous CUDA operations.
	Executing a model comprises potentially dozens of CUDA kernels.  With paged memory,
	copying model weights comprises on the order of a dozen asynchronous memcpys.
	Internally, CUDA has very short queues for managing submitted asynchronous tasks,
	and surprisingly quickly will block ALL asynchronous submissions if there are too
	many outstanding, even those in completely independent streams */
	std::array<cudaEvent_t, MAX_OUTSTANDING_EVENTS> rate_limit_events;

	// Cool
	const Memfile so_memfile;
	std::string serialized_spec;
	int weights_size;
	char* weights_pinned_host_memory; // alloced with cudaMallocHost

	Model(Memfile so_memfile, std::string &serialized_spec, int weights_size, char* weights_pinned_host_memory);

	// Warm
	model::PageMappedModelDef* spec = nullptr;
	unsigned weights_pages_count, workspace_pages_count, total_pages_count;

	std::vector<OpExec>* op_execs = nullptr;
	so::TVMWarmSharedObject* warm_so = nullptr;

	// Hot
	so::TVMHotSharedObject* hot_so = nullptr;
	std::vector<char*>* weights_pages = nullptr;

	// Exec
	std::vector<char*>* workspace_pages = nullptr;

public:
	~Model();

	/* Preconditions: none */
	void instantiate_model_on_host();

	/* Preconditions: instantiate_model_on_host */
	void uninstantiate_model_on_host();

	/* Preconditions: instantiate_model_on_host */
	void instantiate_model_on_device();

	/* Preconditions: instantiate_model_on_device */
	void uninstantiate_model_on_device();

	/* Preconditions: instantiate_model_on_host */
	unsigned num_weights_pages(unsigned page_size);

	/* Preconditions: none */
	void set_weights_pages(std::vector<char*> &weights_pages);

	/* Preconditions: set_weights_pages */
	void unset_weights_pages();

	/* Preconditions: instantiate_model_on_host */
	unsigned num_workspace_pages(unsigned page_size);

	/* Preconditions: none */
	void set_workspace_pages(std::vector<char*> &workspace_pages);

	/* Preconditions: set_workspace_pages */
	void unset_workspace_pages();

	/* Preconditions: set_weights_pages */
	void transfer_weights_to_device(cudaStream_t stream);

	/* Preconditions: instantiate_model_on_host */
	unsigned input_size();

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_input_to_device(char* input_ptr, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_host */
	unsigned output_size();

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_output_from_device(char* output_ptr, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_device */
	void call(cudaStream_t stream);

private:

	char* getpage(unsigned i);
	void make_op_exec(PageMappedOpDef &spec, OpExec &op);
	void call_op_exec(OpExec &op);

public:

	static Model* loadFromDisk(
			std::string so_filename, 
			std::string clockwork_filename,
			std::string clockwork_weights_filename );



};


}
}

#endif