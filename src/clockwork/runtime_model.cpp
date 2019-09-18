#include <dmlc/logging.h>
#include "clockwork/runtime_model.h"

using namespace clockwork;

RuntimeModel::RuntimeModel(PageCache* cache, model::Model* model) : cache(cache), model(model), in_use(ATOMIC_FLAG_INIT) {
	model->instantiate_model_on_host();
	// TODO: remove instantiate_model_on_device and let it be done as part of model exec
	instantiate_code();
}

bool RuntimeModel::try_lock() {
	if (in_use.test_and_set()) return false;
	cache->trylock(weights_pages);
	return true;
}

void RuntimeModel::lock() {
	while (!try_lock()) {}
}

void RuntimeModel::unlock() {
	cache->unlock(weights_pages);
	cache->unlock(workspace_pages);
	cache->free(workspace_pages);
	in_use.clear();
}

bool RuntimeModel::has_code() {
	return instantiated_on_device;
}

void RuntimeModel::instantiate_code() {
	instantiated_on_device = true;
	model->instantiate_model_on_device();
}

void RuntimeModel::uninstantiate_code() {
	if (instantiated_on_device) {
		instantiated_on_device = false;
		model->uninstantiate_model_on_device();
	}
}

bool RuntimeModel::has_weights() {
	return weights_pages != nullptr;
}

void RuntimeModel::evict_weights() {
	cache->unlock(weights_pages);
	cache->free(weights_pages);
}

void RuntimeModel::transfer_weights(cudaStream_t stream) {
	if (weights_pages == nullptr) {
		weights_pages = cache->alloc(model->num_weights_pages(cache->page_size), [this] {
			weights_pages = nullptr;
			model->unset_weights_pages();
		});
		model->set_weights_pages(weights_pages->page_pointers);
	}
	model->transfer_weights_to_device(stream);
}

unsigned RuntimeModel::input_size() {
	return model->input_size();
}

void RuntimeModel::set_input(void* input, cudaStream_t stream) {
	if (workspace_pages == nullptr) {
		workspace_pages = cache->alloc(model->num_workspace_pages(cache->page_size), [this] {
			workspace_pages = nullptr;
			model->unset_workspace_pages();
		});
		model->set_workspace_pages(workspace_pages->page_pointers);
	}
	model->transfer_input_to_device(static_cast<char*>(input), stream);
}

void RuntimeModel::call(cudaStream_t stream) {
	model->call(stream);
}

unsigned RuntimeModel::output_size() {
	return model->output_size();
}

void RuntimeModel::get_output(void* output, cudaStream_t stream) {
	model->transfer_output_from_device(static_cast<char*>(output), stream);
}