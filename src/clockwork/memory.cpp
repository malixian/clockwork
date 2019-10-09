#include "clockwork/memory.h"
#include "tvm/runtime/cuda_common.h"

namespace clockwork {

RuntimeModel::RuntimeModel(model::Model* model) : model(model), in_use(ATOMIC_FLAG_INIT), weights(nullptr), version(0) {
}

bool RuntimeModel::try_lock() {
	return !in_use.test_and_set();
}

void RuntimeModel::lock() {
	while (!try_lock());
}

void RuntimeModel::unlock() {
	in_use.clear();
}


ModelStore::ModelStore() : in_use(ATOMIC_FLAG_INIT) {}

ModelStore::~ModelStore() {
	while (in_use.test_and_set());

	for (auto &p : models) {
		RuntimeModel* rm = p.second;
		if (rm != nullptr) {
			// Do we want to delete models here? Probably?
			delete rm->model;
			delete rm;
		}
	}

	// Let callers hang here to aid in use-after-free
	// in_use.clear();
}

RuntimeModel* ModelStore::get(int model_id) {
	while (in_use.test_and_set());

	RuntimeModel* rm = models[model_id];

	in_use.clear();

	return rm;
}

bool ModelStore::contains(int model_id) {
	while (in_use.test_and_set());

	RuntimeModel* rm = models[model_id];

	in_use.clear();

	return rm != nullptr;
}

void ModelStore::put(int model_id, RuntimeModel* model) {
	while (in_use.test_and_set());

	models[model_id] = model;

	in_use.clear();
}

bool ModelStore::put_if_absent(int model_id, RuntimeModel* model) {
	while (in_use.test_and_set());

	bool did_put = false;
	if (models[model_id] == nullptr) {
		models[model_id] = model;
		did_put = true;
	}

	in_use.clear();

	return did_put;
}

CUDAPageCache::CUDAPageCache(char* baseptr, uint64_t total_size, uint64_t page_size, const bool allowEvictions) :
		PageCache(baseptr, total_size, page_size, allowEvictions) {
}

CUDAPageCache::~CUDAPageCache() {
	CUDA_CALL(cudaFree(this->baseptr));
}

PageCache* make_GPU_cache(size_t cache_size, size_t page_size) {
	cache_size = page_size * (cache_size / page_size);
	void* baseptr;
	CUDA_CALL(cudaMalloc(&baseptr, cache_size));
	return new CUDAPageCache(static_cast<char*>(baseptr), cache_size, page_size, false);
}

}