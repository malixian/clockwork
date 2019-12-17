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

// MemoryManager::MemoryManager(MemoryPool* host_io_pool, 
// 			PageCache* device_weights_cache, 
// 			MemoryPool* device_workspace_pool, 
// 			MemoryPool* device_io_pool) : 
// 		models(new ModelStore()), host_io_pool(host_io_pool),
// 		device_weights_cache(device_weights_cache), device_workspace_pool(device_workspace_pool),
// 		device_io_pool(device_io_pool) {}

// MemoryManager::~MemoryManager() {
// 	delete models;
// 	delete host_io_pool;
// 	delete device_weights_cache;
// 	delete device_workspace_pool;
// 	delete device_io_pool;
// }

MemoryManager::MemoryManager(PageCache* weights_cache, PageCache* workspace_cache) : 
			weights_cache(weights_cache), 
			workspace_cache(workspace_cache), 
			models(new ModelStore()) {
	this->io_cache = make_IO_cache();
}

MemoryManager::~MemoryManager() {
	delete models;
	if (weights_cache != workspace_cache) {
		delete weights_cache;
	}
	delete workspace_cache;
	delete io_cache;
}


IOCache::IOCache(size_t total_size, size_t page_size) : page_size(page_size) {
	total_size = page_size * (total_size / page_size);
	CUDA_CALL(cudaMallocHost(&baseptr, total_size));
	for (size_t offset = 0; offset < total_size; offset += page_size) {
		ptrs.push(baseptr + offset);
	}
}

IOCache::~IOCache() {
	CUDA_CALL(cudaFreeHost(baseptr));
}

bool IOCache::take(char* &ptr) {
	return ptrs.try_pop(ptr);
}

void IOCache::release(char* ptr) {
	ptrs.push(ptr);
}



MemoryPool::MemoryPool(char* base_ptr, size_t size) : base_ptr(base_ptr), size(size) {
}

MemoryPool::~MemoryPool() {}

// Allocate `amount` of memory; returns nullptr if out of memory
std::shared_ptr<MemoryAllocation> MemoryPool::alloc(size_t amount) {
	std::lock_guard<std::mutex> lock(mutex);

	// Simple case when there are no outstanding allocations
	if (allocations.size() == 0) {
		if (amount > size) return nullptr; // Too big for the pool

		auto allocation = std::make_shared<MemoryAllocation>(base_ptr, 0, amount);
		allocations.push_back(allocation);
		return allocation;
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
			return allocation;
		}

		if (amount <= front->offset) {
			// Fits in pool

			auto allocation = std::make_shared<MemoryAllocation>(base_ptr, 0, amount);
			allocations.push_back(allocation);
			return allocation;
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
			return allocation;
		}

		// Doesn't fit in pool
		return nullptr;
	}
}

// Return the memory back to the pool
void MemoryPool::free(std::shared_ptr<MemoryAllocation> &allocation) {
	std::lock_guard<std::mutex> lock(mutex);

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

IOCache* make_IO_cache() {
        // TODO: don't hard-code
       size_t cache_size = 512L * 1024L * 1024L;
       size_t page_size = 64L * 1024L * 1024L;
       return new IOCache(cache_size, page_size);
}

}