#ifndef _CLOCKWORK_MODEL_BATCHED_H_
#define _CLOCKWORK_MODEL_BATCHED_H_

#include "clockwork/model/model.h"

namespace clockwork {
namespace model {

class BatchedModel {
public:
	std::vector<Model*> model_lookup;
	std::vector<std::pair<unsigned, Model*>> models;

	int single_input_size;
	int single_output_size;

	int weights_size;
	char* weights_pinned_host_memory; // alloced with cudaMallocHost

	BatchedModel(int weights_size, char* weights_pinned_host_memory, std::vector<std::pair<unsigned, Model*>> models);

public:
	virtual ~BatchedModel();

	/* Preconditions: none */
	void instantiate_models_on_host();

	bool is_valid_batch_size(unsigned batch_size);
	void check_batch_size(unsigned batch_size);

	/* Preconditions: instantiate_model_on_host */
	void uninstantiate_models_on_host();

	/* Preconditions: instantiate_model_on_host */
	void instantiate_models_on_device();

	/* Preconditions: instantiate_model_on_device */
	void uninstantiate_models_on_device();

	/* The actual batch size implementations */
	std::vector<unsigned> implemented_batch_sizes();

	unsigned padded_batch_size(unsigned batch_size);
	unsigned max_batch_size();

	/* Preconditions: instantiate_model_on_host */
	unsigned num_weights_pages(unsigned page_size);
	size_t workspace_memory_size(unsigned batch_size);
	size_t io_memory_size(unsigned batch_size);

	/* Preconditions: set_weights_pages */
	void transfer_weights_to_device(std::vector<char*> &weights_pages, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_host */
	size_t input_size(unsigned batch_size);
	size_t input_size_with_padding(unsigned batch_size);

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_input_to_device(unsigned batch_size, const char* input_ptr, char* &dst_io_memory, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_host */
	size_t output_size(unsigned batch_size);
	size_t output_size_with_padding(unsigned batch_size);

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_output_from_device(unsigned batch_size, char* output_ptr, char* &src_io_memory, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_device */
	void call(unsigned batch_size, std::vector<char*> &weights_pages, char* &io_memory, char* &workspace_memory, cudaStream_t stream);

public:

	static BatchedModel* loadFromDisk(std::string base_filename);


};

}
}

#endif