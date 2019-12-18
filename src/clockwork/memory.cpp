#include "clockwork/memory.h"
#include "tvm/runtime/cuda_common.h"

namespace clockwork {

RuntimeModel::RuntimeModel(model::BatchedModel* model) : model(model), in_use(ATOMIC_FLAG_INIT), weights(nullptr), version(0) {
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

MemoryManager::MemoryManager(
		size_t weights_cache_size, size_t weights_cache_page_size,
		size_t io_pool_size,
		size_t workspace_pool_size,
		size_t host_io_pool_size) :
			weights_cache(make_GPU_cache(weights_cache_size, weights_cache_page_size)),
			io_pool(CUDAMemoryPool::create(io_pool_size)),
			workspace_pool(CUDAMemoryPool::create(workspace_pool_size)),
			host_io_pool(CUDAHostMemoryPool::create(host_io_pool_size)),
			models(new ModelStore()) {
}

MemoryManager::~MemoryManager() {
	delete models;
	delete weights_cache;
	delete io_pool;
	delete workspace_pool;
	delete host_io_pool;
}


MemoryPool::MemoryPool(char* base_ptr, size_t size) : base_ptr(base_ptr), size(size) {
}

MemoryPool::~MemoryPool() {}

// Allocate `amount` of memory; returns nullptr if out of memory
char* MemoryPool::alloc(size_t amount) {
	std::lock_guard<std::mutex> lock(mutex);

	// Simple case when there are no outstanding allocations
	if (allocations.size() == 0) {
		if (amount > size) return nullptr; // Too big for the pool

		auto allocation = std::make_shared<MemoryAllocation>(base_ptr, 0, amount);
		allocations.push_back(allocation);
		ptr_allocations[base_ptr] = allocation;
		return base_ptr;
	}

	auto front = allocations.front();
	auto back = allocations.back();

	if (front->offset <= back->offset) {
		// Case where memory is one contiguous range

		size_t offset = back->offset + back->size;
		if (offset + amount <= size) {
			// Fits in pool

			auto allocation = std::make_shared<MemoryAllocation>(base_ptr, offset, amount);
			allocations.push_back(allocation);
			ptr_allocations[base_ptr + offset] = allocation;
			return base_ptr + offset;
		}

		if (amount <= front->offset) {
			// Fits in pool

			auto allocation = std::make_shared<MemoryAllocation>(base_ptr, 0, amount);
			allocations.push_back(allocation);
			ptr_allocations[base_ptr] = allocation;
			return base_ptr;
		}

		// Doesn't fit in pool
		return nullptr;

	} else {
		// Case where memory wraps round

		size_t offset = back->offset + back->size;
		if (offset + amount <= front->offset) {
			// Fits in pool

			auto allocation = std::make_shared<MemoryAllocation>(base_ptr, offset, amount);
			allocations.push_back(allocation);
			ptr_allocations[base_ptr + offset] = allocation;
			return base_ptr + offset;
		}

		// Doesn't fit in pool
		return nullptr;
	}
}

// Return the memory back to the pool
void MemoryPool::free(char* ptr) {
	std::lock_guard<std::mutex> lock(mutex);

	auto it = ptr_allocations.find(ptr);
	if (it == ptr_allocations.end()) return;

	auto allocation = it->second;

	ptr_allocations.erase(it);

	allocation->freed.store(true);

	// Pop all freed allocations from the queue
	while (allocations.size() > 0 && allocations.front()->freed) {
		allocations.pop_front();
	}
}

CUDAMemoryPool::CUDAMemoryPool(char* base_ptr, size_t size) : MemoryPool(base_ptr, size) {}

CUDAMemoryPool::~CUDAMemoryPool() {
	CUDA_CALL(cudaFree(base_ptr));
}

CUDAMemoryPool* CUDAMemoryPool::create(size_t size) {
	void* baseptr;
	CUDA_CALL(cudaMalloc(&baseptr, size));
	return new CUDAMemoryPool(static_cast<char*>(baseptr), size);
}

CUDAHostMemoryPool::CUDAHostMemoryPool(char* base_ptr, size_t size) : MemoryPool(base_ptr, size) {}

CUDAHostMemoryPool::~CUDAHostMemoryPool() {
	CUDA_CALL(cudaFreeHost(base_ptr));
}

CUDAHostMemoryPool* CUDAHostMemoryPool::create(size_t size) {
	void* baseptr;
	CUDA_CALL(cudaMallocHost(&baseptr, size));
	return new CUDAHostMemoryPool(static_cast<char*>(baseptr), size);	
}

}