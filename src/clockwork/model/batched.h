#include <algorithm>
#include "clockwork/model/model.h"

namespace clockwork {
namespace model {

class BatchedModel {
public:
	std::vector<Model*> model_lookup;
	std::vector<std::pair<unsigned, Model*>> models;

	int weights_size;
	char* weights_pinned_host_memory; // alloced with cudaMallocHost

	Model(int weights_size, char* weights_pinned_host_memory, std::vector<std::pair<unsigned, Model*>> models):
			weights_size(weights_size), weights_pinned_host_memory(weights_pinned_host_memory), models(models) {
		std::sort(models.begin(), models.end());

		unsigned batch_size = 0;
		for (auto &p : models) {
			while (batch_size < p.first) {
				model_lookup.push_back(p.second);
				batch_size++;
			}
		}
	}

public:
	virtual ~BatchedModel();

	/* Preconditions: none */
	void instantiate_models_on_host() {
		for (auto &p : models) {
			p.second->instantiate_model_on_host();
		}

		// Perform checks
		unsigned expected_input_size = 0;
		unsigned expected_output_size = 0;
		for (auto &p : models) {
			unsigned batch_size = p.first;
			Model* model = p.second;

			unsigned single_input_size = model->input_size() / batch_size;

			if (expected_input_size == 0) {
				expected_input_size = single_input_size;
			} else {
				CHECK(expected_input_size == single_input_size) 
					<< "Inconsistent input sizes between batch variants " 
					<< expected_input_size << " and " << single_input_size;
			}

			unsigned single_output_size = model->input_size() / batch_size;

			if (expected_output_size == 0) {
				expected_output_size = single_output_size;
			} else {
				CHECK(expected_output_size == single_output_size) 
					<< "Inconsistent output sizes between batch variants " 
					<< expected_output_size << " and " << single_output_size;
			}
		}
	}

	void check_batch_size(unsigned batch_size) {
		CHECK(batch_size < model_lookup.size()) << "Unsupported batch size " << batch_size << " larger than maximum " << model_lookup.size();
	}

	/* Preconditions: instantiate_model_on_host */
	void uninstantiate_models_on_host() {
		for (auto &p : models) {
			p.second->uninstantiate_model_on_host();
		}
	}

	/* Preconditions: instantiate_model_on_host */
	void instantiate_models_on_device() {
		for (auto &p : models) {
			p.second->instantiate_model_on_device();
		}
	}

	/* Preconditions: instantiate_model_on_device */
	void uninstantiate_models_on_device() {
		for (auto &p : models) {
			p.second->uninstantiate_model_on_device();
		}
	}

	/* Preconditions: instantiate_model_on_host */
	unsigned num_weights_pages(unsigned page_size) {
		return model_lookup[0]->num_weights_pages(page_size);
	}

	/* Preconditions: instantiate_model_on_host */
	unsigned num_workspace_pages(unsigned batch_size, unsigned page_size) {
		check_batch_size(batch_size);
		return model_lookup[batch_size]->num_workspace_pages(page_size);
	}

	/* Preconditions: set_weights_pages */
	void transfer_weights_to_device(std::vector<char*> &weights_pages, cudaStream_t stream) {
		model_lookup[0]->transfer_weights_to_device(weights_pages, stream);
	}

	/* Preconditions: instantiate_model_on_host */
	unsigned input_size(unsigned batch_size) {
		check_batch_size(batch_size);
		return model_lookup[batch_size]->input_size();
	}

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_input_to_device(unsigned batch_size, const char* input_ptr, std::vector<char*> &workspace_pages, cudaStream_t stream) {
		check_batch_size(batch_size);
		model_lookup[batch_size]->transfer_input_to_device(input_ptr, workspace_pages, stream);
	}

	/* Preconditions: instantiate_model_on_host */
	unsigned output_size(unsigned batch_size) {
		check_batch_size(batch_size);
		return model_lookup[batch_size]->output_size();
	}

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_output_from_device(unsigned batch_size, char* output_ptr, std::vector<char*> &workspace_pages, cudaStream_t stream) {
		check_batch_size(batch_size);
		model_lookup[batch_size]->transfer_output_from_device(output_ptr, workspace_pages, stream);		
	}

	/* Preconditions: instantiate_model_on_device */
	void call(unsigned batch_size, std::vector<char*> &weights_pages, std::vector<char*> &workspace_pages, cudaStream_t stream) {
		check_batch_size(batch_size);
		model_lookup[batch_size]->call(weights_pages, workspace_pages, stream);
	}

public:

	static Model* loadFromDisk(std::string base_filename) {
		// TODO
		throw "TODO";
	}


};

}
}