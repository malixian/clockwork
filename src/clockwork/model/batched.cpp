#include "clockwork/model/batched.h"

#include <algorithm>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/model/model.h"

namespace clockwork {
namespace model {

BatchedModel::BatchedModel(int weights_size, char* weights_pinned_host_memory, std::vector<std::pair<unsigned, Model*>> models):
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

BatchedModel::~BatchedModel() {}

void BatchedModel::instantiate_models_on_host() {
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

		unsigned single_output_size = model->output_size() / batch_size;

		if (expected_output_size == 0) {
			expected_output_size = single_output_size;
		} else {
			CHECK(expected_output_size == single_output_size) 
				<< "Inconsistent output sizes between batch variants " 
				<< expected_output_size << " and " << single_output_size;
		}
	}

	this->single_input_size = expected_input_size;
	this->single_output_size = expected_output_size;
}

void BatchedModel::check_batch_size(unsigned batch_size) {
	CHECK(batch_size < model_lookup.size()) << "Unsupported batch size " << batch_size << " larger than maximum " << model_lookup.size();
}

void BatchedModel::uninstantiate_models_on_host() {
	for (auto &p : models) {
		p.second->uninstantiate_model_on_host();
	}
}

void BatchedModel::instantiate_models_on_device() {
	for (auto &p : models) {
		p.second->instantiate_model_on_device();
	}
}

void BatchedModel::uninstantiate_models_on_device() {
	for (auto &p : models) {
		p.second->uninstantiate_model_on_device();
	}
}

std::vector<unsigned> BatchedModel::implemented_batch_sizes() {
	std::vector<unsigned> sizes(models.size());
	for (unsigned i = 0; i < models.size(); i++) {
		sizes[i] = models[i].first;
	}
	return sizes;
}

unsigned BatchedModel::max_batch_size() {
	return models[models.size()-1].first;
}

unsigned BatchedModel::padded_batch_size(unsigned batch_size) {
	check_batch_size(batch_size);
	for (auto &p : models) {
		if (batch_size < p.first) {
			return p.first;
		}
	}
}

unsigned BatchedModel::num_weights_pages(unsigned page_size) {
	return model_lookup[0]->num_weights_pages(page_size);
}

unsigned BatchedModel::num_workspace_pages(unsigned batch_size, unsigned page_size) {
	check_batch_size(batch_size);
	return model_lookup[batch_size]->num_workspace_pages(page_size);
}

void BatchedModel::transfer_weights_to_device(std::vector<char*> &weights_pages, cudaStream_t stream) {
	model_lookup[0]->transfer_weights_to_device(weights_pages, stream);
}

unsigned BatchedModel::input_size(unsigned batch_size) {
	check_batch_size(batch_size);
	return single_input_size * batch_size;
}

unsigned BatchedModel::input_size_with_padding(unsigned batch_size) {
	check_batch_size(batch_size);
	return model_lookup[batch_size]->input_size();
}

void BatchedModel::transfer_input_to_device(unsigned batch_size, const char* input_ptr, std::vector<char*> &workspace_pages, cudaStream_t stream) {
	check_batch_size(batch_size);
	model_lookup[batch_size]->transfer_input_to_device(input_ptr, workspace_pages, stream);
}

unsigned BatchedModel::output_size(unsigned batch_size) {
	check_batch_size(batch_size);
	return single_output_size * batch_size;
}

unsigned BatchedModel::output_size_with_padding(unsigned batch_size) {
	check_batch_size(batch_size);
	return model_lookup[batch_size]->output_size();
}

void BatchedModel::transfer_output_from_device(unsigned batch_size, char* output_ptr, std::vector<char*> &workspace_pages, cudaStream_t stream) {
	check_batch_size(batch_size);
	model_lookup[batch_size]->transfer_output_from_device(output_ptr, workspace_pages, stream);		
}

void BatchedModel::call(unsigned batch_size, std::vector<char*> &weights_pages, std::vector<char*> &workspace_pages, cudaStream_t stream) {
	check_batch_size(batch_size);
	model_lookup[batch_size]->call(weights_pages, workspace_pages, stream);
}

BatchedModel* BatchedModel::loadFromDisk(std::string base_filename) {
	std::string clockwork_weights_filename = base_filename + ".clockwork_params";

	// Load shared weights
	std::string weights;
	util::readFileAsString(clockwork_weights_filename, weights);
	int weights_size = weights.size();
	char* weights_pinned_host_memory;
	CUDA_CALL(cudaMallocHost(&weights_pinned_host_memory, weights_size));
	std::memcpy(weights_pinned_host_memory, weights.data(), weights_size);

	std::vector<std::pair<unsigned, Model*>> models;

	int batchsize = 1;
	while (true) {
		std::stringstream batch_filename_base;
		batch_filename_base << base_filename << "." << batchsize;

		std::string so_filename = batch_filename_base.str() + ".so";
		std::string clockwork_filename = batch_filename_base.str() + ".clockwork";

		if (!util::exists(so_filename) || !util::exists(clockwork_filename)) {
			break;
		}

		Memfile so_memfile = Memfile::readFrom(so_filename);

		std::string clockwork_serialized_spec;
		util::readFileAsString(clockwork_filename, clockwork_serialized_spec);

		Model* model = new Model(so_memfile, clockwork_serialized_spec, weights_size, weights_pinned_host_memory);
		models.push_back(std::make_pair(batchsize, model));

		batchsize *= 2;
	}

	return new BatchedModel(weights_size, weights_pinned_host_memory, models);
}

}
}