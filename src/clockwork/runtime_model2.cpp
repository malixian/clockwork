#include <dmlc/logging.h>
#include "clockwork/runtime_model.h"

using namespace clockwork;

RuntimeModel2::RuntimeModel2(PageCache* cache, Model* model) : cache(cache), model(model), in_use(ATOMIC_FLAG_INIT) {
	params_callback = new ParamsEvictionCallback(this);
	workspace_callback = new WorkspaceEvictionCallback(this);
	model->instantiate_model_on_host();
}

bool RuntimeModel2::try_lock() {
	if (in_use.test_and_set()) return false;
	cache->trylock(weights_pages);
	return true;
}

void RuntimeModel2::lock() {
	while (!try_lock()) {}
}

void RuntimeModel2::unlock() {
	cache->unlock(weights_pages);
	cache->free(workspace_pages);
	in_use.clear();
}

bool RuntimeModel2::has_code() {
	return instantiated_on_device;
}

void RuntimeModel2::instantiate_code() {
	instantiated_on_device = true;
	model->instantiate_model_on_device();
}

void RuntimeModel2::uninstantiate_code() {
	if (instantiated_on_device) {
		instantiated_on_device = false;
		model->uninstantiate_model_on_device();
	}
}

bool RuntimeModel2::has_weights() {
	return weights_pages != nullptr;
}

void RuntimeModel2::evict_weights() {
	cache->unlock(weights_pages);
	cache->free(weights_pages);
}

void RuntimeModel2::transfer_weights(cudaStream_t stream) {
	if (weights_pages == nullptr) {
		weights_pages = cache->alloc(model->num_weights_pages(cache->page_size), weights_evicted);
		model->set_weights_pages(weights_pages->page_pointers);
	}
	model->transfer_weights_to_device(stream);
}

unsigned RuntimeModel2::input_size() {
	return model->input_size();
}

void RuntimeModel2::set_input(void* input, cudaStream_t stream) {
	if (workspace_pages == nullptr) {
		workspace_pages = cache->alloc(model->num_workspace_pages(cache->page_size), workspace_evicted);
		model->set_workspace_pages(workspace_pages->page_pointers);
	}
	model->transfer_input_to_device(input, stream);
}

void RuntimeModel2::call(cudaStream_t stream) {
	model->call(stream);
}

unsigned RuntimeModel2::output_size() {
	return model->output_size();
}

void RuntimeModel2::get_output(void* output, cudaStream_t stream) {
	model->transfer_output_from_device(output, stream);
}