#ifndef _CLOCKWORK_MODEL_H_
#define _CLOCKWORK_MODEL_H_

#include <string>
#include <array>
#include "clockwork/modeldef.h"
#include "clockwork/model/memfile.h"
#include "clockwork/model/so.h"
#include <cuda_runtime.h>
#include "clockwork/util.h"

#define MAX_OUTSTANDING_EVENTS 16
#define MAX_OUTSTANDING_EXEC_EVENTS 16
#define MAX_OUTSTANDING_MEMCPY_EVENTS 2

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
public:
	unsigned gpu_id;

	// Cool
	const Memfile so_memfile;
	std::string serialized_spec;
	int weights_size;
	char* weights_pinned_host_memory; // alloced with cudaMallocHost

	Model(Memfile so_memfile, std::string &serialized_spec, int weights_size,
		char* weights_pinned_host_memory, unsigned gpu_id);


private:

	/* These events are used to rate-limit submission of asynchronous CUDA operations.
	Executing a model comprises potentially dozens of CUDA kernels.  With paged memory,
	copying model weights comprises on the order of a dozen asynchronous memcpys.
	Internally, CUDA has very short queues for managing submitted asynchronous tasks,
	and surprisingly quickly will block ALL asynchronous submissions if there are too
	many outstanding, even those in completely independent streams */
	std::array<cudaEvent_t, MAX_OUTSTANDING_EVENTS> rate_limit_events;


	// Warm
	model::PageMappedModelDef* spec = nullptr;
	unsigned weights_pages_count;
	size_t io_size, workspace_size;

	std::vector<OpExec>* op_execs = nullptr;
	so::TVMWarmSharedObject* warm_so = nullptr;

	// Hot
	so::TVMHotSharedObject* hot_so = nullptr;

public:
	virtual ~Model();

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
	size_t workspace_memory_size();
	size_t io_memory_size();

	/* Preconditions: set_weights_pages */
	void transfer_weights_to_device(std::vector<char*> &weights_pages, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_host */
	size_t input_size();

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_input_to_device(const char* input_ptr, char* &dst_io_memory, cudaStream_t stream);
	void transfer_input_to_device(size_t input_size, const char* input_ptr, char* &dst_io_memory, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_host */
	size_t output_size();

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_output_from_device(char* output_ptr, char* &src_io_memory, cudaStream_t stream);
	void transfer_output_from_device(size_t output_size, char* output_ptr, char* &src_io_memory, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_device */
	void call(std::vector<char*> &weights_pages, char* &io_memory, char* &workspace_memory, cudaStream_t stream);

private:

	void make_op_exec(PageMappedOpDef &spec, OpExec &op);
	void call_op_exec(OpExec &op, std::vector<char*> &pages);

public:

	static Model* loadFromDisk(
			std::string so_filename, 
			std::string clockwork_filename,
			std::string clockwork_weights_filename,
			unsigned gpu_id);



};


class DiskModel : public Model {
public:
	DiskModel(Memfile so_memfile, std::string &serialized_spec, int weights_size,
		char* weights_pinned_host_memory, unsigned gpu_id);
	~DiskModel();
};


}
}

#endif