#include "clockwork/model/batched.h"

#include <algorithm>
#include "clockwork/cuda_common.h"
#include "clockwork/model/model.h"
#include <libconfig.h++>
#include <sstream>

namespace clockwork {
namespace model {

BatchedModel::BatchedModel(int weights_size, char* weights_pinned_host_memory,
	std::vector<std::pair<unsigned, Model*>> models, unsigned gpu_id, std::string source):
		weights_size(weights_size),
		weights_pinned_host_memory(weights_pinned_host_memory),
		models(models),
		gpu_id(gpu_id),
		source(source) {
	std::sort(models.begin(), models.end());

	unsigned batch_size = 0;
	for (auto &p : models) {
		while (batch_size <= p.first) {
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
				<< "b=" << batch_size << " has  " << single_input_size << " per input, expected " << expected_input_size;
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

bool BatchedModel::is_valid_batch_size(unsigned batch_size) {
	return batch_size < model_lookup.size();
}

void BatchedModel::check_batch_size(unsigned batch_size) {
	CHECK(batch_size < model_lookup.size()) << "Unsupported batch size " << batch_size << " larger than maximum " << (model_lookup.size()-1);
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
		if (batch_size <= p.first) {
			return p.first;
		}
	}
	return models[models.size() - 1].first;
}

unsigned BatchedModel::num_weights_pages(unsigned page_size) {
	return model_lookup[0]->num_weights_pages(page_size);
}

size_t BatchedModel::workspace_memory_size(unsigned batch_size) {
	check_batch_size(batch_size);
	return model_lookup[batch_size]->workspace_memory_size();
}

size_t BatchedModel::io_memory_size(unsigned batch_size) {
	check_batch_size(batch_size);
	return model_lookup[batch_size]->io_memory_size();
}

void BatchedModel::transfer_weights_to_device(std::vector<char*> &weights_pages, cudaStream_t stream) {
	model_lookup[0]->transfer_weights_to_device(weights_pages, stream);
}

size_t BatchedModel::input_size(unsigned batch_size) {
	check_batch_size(batch_size);
	return single_input_size * batch_size;
}

size_t BatchedModel::input_size_with_padding(unsigned batch_size) {
	check_batch_size(batch_size);
	return model_lookup[batch_size]->input_size();
}

void BatchedModel::transfer_input_to_device(unsigned batch_size, const char* input_ptr, char* &dst_io_memory, cudaStream_t stream) {
	check_batch_size(batch_size);
	model_lookup[batch_size]->transfer_input_to_device(single_input_size * batch_size, input_ptr, dst_io_memory, stream);
}

size_t BatchedModel::output_size(unsigned batch_size) {
	check_batch_size(batch_size);
	return single_output_size * batch_size;
}

size_t BatchedModel::output_size_with_padding(unsigned batch_size) {
	check_batch_size(batch_size);
	return model_lookup[batch_size]->output_size();
}

void BatchedModel::transfer_output_from_device(unsigned batch_size, char* output_ptr, char* &src_io_memory, cudaStream_t stream) {
	check_batch_size(batch_size);
	model_lookup[batch_size]->transfer_output_from_device(single_output_size * batch_size, output_ptr, src_io_memory, stream);		
}

void BatchedModel::call(unsigned batch_size, std::vector<char*> &weights_pages, char* &io_memory, char* &workspace_memory, cudaStream_t stream) {
	check_batch_size(batch_size);
	model_lookup[batch_size]->call(weights_pages, io_memory, workspace_memory, stream);
}

void lookupValue(libconfig::Config &config, std::string key, uint64_t &value) {
	unsigned long long v = 0;
	if (config.getRoot().lookupValue(key, v)) {
		value = v;
	}
}

BatchedModel* BatchedModel::loadFromDisk(std::string base_filename, unsigned gpu_id) {
	std::string clockwork_weights_filename = base_filename + ".clockwork_params";

	// Load shared weights
	std::string weights;
	util::readFileAsString(clockwork_weights_filename, weights);
	int weights_size = weights.size();
	char* weights_pinned_host_memory;
	CUDA_CALL(cudaSetDevice(gpu_id)); // TODO Is this really needed?
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

		Model* model = new Model(so_memfile, clockwork_serialized_spec, weights_size, weights_pinned_host_memory, gpu_id);
		models.push_back(std::make_pair(batchsize, model));

		batchsize *= 2;
	}

	CHECK(batchsize != 1) << "No valid batch sizes found for " << base_filename;

	auto batched = new BatchedModel(weights_size, weights_pinned_host_memory, models, gpu_id, base_filename);

	// Lastly, load measurements if they exist
	try {
		std::string measurements_file = base_filename + ".measurements";
		libconfig::Config measurements;
		measurements.readFile(measurements_file.c_str());

		lookupValue(measurements, "weights", batched->transfer_measurement);
		for (auto p : models) {
			std::stringstream key;
			key << "b" << p.first;
			lookupValue(measurements, key.str(), p.second->exec_measurement);
		}
	} catch (const libconfig::FileIOException& e) {
		// No measurements file; just ignore and move on
	}

	return batched;
}

std::vector<BatchedModel*> BatchedModel::loadMultipleFromDisk(std::string base_filename, unsigned gpu_id, int num_copies) {

	std::vector<BatchedModel*> batched_models;
	BatchedModel* loaded_model = BatchedModel::loadFromDisk(base_filename, gpu_id);
	batched_models.push_back(loaded_model);

	if ( num_copies > 1) {
		std::vector<std::pair<unsigned, model::Model*>> models = loaded_model->models;

		void* mega_memory;
		size_t mega_size = (size_t) loaded_model->weights_size * (num_copies - 1);
		CUDA_CALL(cudaMallocHost(&mega_memory, mega_size));

		int copies_so_far = 0;
		for (int id = 1; id < num_copies; id++) {

			std::vector<std::pair<unsigned, model::Model*>> duplicate_models;

			size_t offset = (size_t) loaded_model->weights_size * copies_so_far;
			char* weights_pinned_host_memory = static_cast<char*>(mega_memory) + offset;
			std::memcpy(weights_pinned_host_memory, loaded_model->weights_pinned_host_memory, loaded_model->weights_size);
			copies_so_far ++;

			for (auto &p : models) {

				Memfile so_memfile = Memfile::readFrom(p.second->so_memfile.filename);
				std::string serialized_spec = p.second->serialized_spec;

				model::Model* model = new model::Model(so_memfile, serialized_spec, p.second->weights_size,  static_cast<char*>(weights_pinned_host_memory), gpu_id);
				model->exec_measurement = p.second->exec_measurement;

				duplicate_models.push_back(std::make_pair(p.first, model));
			}

			model::BatchedModel* duplicate_model =  new model::BatchedModel( loaded_model->weights_size, static_cast<char*>(weights_pinned_host_memory), duplicate_models, gpu_id);
			duplicate_model->transfer_measurement = loaded_model->transfer_measurement;


			batched_models.push_back(duplicate_model);

		}
	}
	return batched_models;
}

}
}
