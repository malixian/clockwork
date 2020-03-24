#include "clockwork/memory.h"
#include "tvm/runtime/cuda_common.h"

namespace clockwork {

RuntimeModel::RuntimeModel(model::BatchedModel* model, unsigned gpu_id):
	model(model), gpu_id(gpu_id), in_use(ATOMIC_FLAG_INIT), weights(nullptr), version(0) {
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

RuntimeModel* ModelStore::get(int model_id, unsigned gpu_id) {
	while (in_use.test_and_set());

	RuntimeModel* rm = models[std::make_pair(model_id, gpu_id)];

	in_use.clear();

	return rm;
}

bool ModelStore::contains(int model_id, unsigned gpu_id) {
	while (in_use.test_and_set());

	RuntimeModel* rm = models[std::make_pair(model_id, gpu_id)];

	in_use.clear();

	return rm != nullptr;
}

void ModelStore::put(int model_id, unsigned gpu_id, RuntimeModel* model) {
	while (in_use.test_and_set());

	models[std::make_pair(model_id, gpu_id)] = model;

	in_use.clear();
}

bool ModelStore::put_if_absent(int model_id, unsigned gpu_id, RuntimeModel* model) {
	while (in_use.test_and_set());

	bool did_put = false;
	std::pair<int, unsigned> key = std::make_pair(model_id, gpu_id);
	if (models[key] == nullptr) {
		models[key] = model;
		did_put = true;
	}

	in_use.clear();

	return did_put;
}

void MemoryManager::initialize(ClockworkWorkerSettings settings) {
	for (unsigned gpu_id = 0; gpu_id < settings.num_gpus; gpu_id++) {
		weights_caches.push_back(make_GPU_cache(settings.weights_cache_size, settings.weights_cache_page_size, gpu_id));
		workspace_pools.push_back(CUDAMemoryPool::create(settings.workspace_pool_size, gpu_id));
		io_pools.push_back(CUDAMemoryPool::create(settings.io_pool_size, gpu_id));
	}
}

MemoryManager::MemoryManager(ClockworkWorkerSettings settings) :
			host_io_pool(CUDAHostMemoryPool::create(settings.host_io_pool_size)),
			models(new ModelStore()),
			num_gpus(settings.num_gpus) {

	initialize(settings);
}

MemoryManager::~MemoryManager() {
	delete models;
	delete host_io_pool;
	for (unsigned i = 0; i < num_gpus; i++) {
		delete weights_caches[i];
		delete workspace_pools[i];
		delete io_pools[i];
	}
}

void MemoryManager::get_worker_memory_info(workerapi::WorkerMemoryInfo &worker_memory_info) {
	// For weights cachs, IO pool, and workspace pool, which are replicated to
	// support multiple GPUs, we return the total (and total remaining) size
	// of the respective cache/pool across all GPUs
	worker_memory_info.weights_cache_total = 0;
	worker_memory_info.weights_cache_remaining = 0;
	worker_memory_info.io_pool_total = 0;
	worker_memory_info.io_pool_remaining = 0;
	worker_memory_info.workspace_pool_total = 0;
	worker_memory_info.workspace_pool_remaining = 0;
	for (unsigned i = 0; i < num_gpus; i++) {
		worker_memory_info.weights_cache_total += weights_caches[i]->size;
		worker_memory_info.weights_cache_remaining += weights_caches[i]->page_size * weights_caches[i]->freePages.size();
		worker_memory_info.io_pool_total += io_pools[i]->size;
		worker_memory_info.io_pool_remaining += io_pools[i]->remaining();
		worker_memory_info.workspace_pool_total += workspace_pools[i]->size;
		worker_memory_info.workspace_pool_remaining += workspace_pools[i]->remaining();
	}

	while (models->in_use.test_and_set());

	// assuming models for different GPUs have the same weights_size,
	// only returning the weights size of the model corresponding to GPU 0 here

	for (auto it = models->models.begin(); it != models->models.end(); ++it) {
		// Note that the key used in models is a pair (model ID, GPU ID)
		// Thus, std::get<0>(it->first) returns the model ID and
		// std::get<1>(it->first) returns the GPU ID

		if (std::get<1>(it->first) != 0) { continue; }
		workerapi::ModelInfo model;
		model.id = std::get<0>(it->first);
		RuntimeModel *runtime_model = it->second;
		model::BatchedModel* batched_model = runtime_model->model;
		model.size = batched_model->weights_size;
		worker_memory_info.models.push_back(model);
	}

	models->in_use.clear();
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

// Get the  size of all allocations
size_t MemoryPool::remaining() {
	size_t allocated = 0;
	for (unsigned i = 0; i < allocations.size(); i++) {
		allocated += allocations[i]->size;
	}
	return (size - allocated);
}

// Reclaim back all allocations
void MemoryPool::clear() {
	std::lock_guard<std::mutex> lock(mutex);

    /* Not really needed
    // Set all allocations pointed to by ptrs in ptr_allocations to "freed"
    for (auto it = ptr_allocations.begin(); it != ptr_allocations.end(); it++) {
        auto allocation = it->second;
        allocation->freed.store(true);
    }

    // Pop all freed allocations from the queue
    while (allocations.size() > 0 && allocations.front()->freed) {
        allocations.pop_front();
    } */

    // Clear the ptr_allocations map
    ptr_allocations.clear();

    // Clear the allocations deque
    allocations.clear();
}

CUDAMemoryPool::CUDAMemoryPool(char* base_ptr, size_t size, unsigned gpu_id):
	MemoryPool(base_ptr, size), gpu_id(gpu_id) {}

CUDAMemoryPool::~CUDAMemoryPool() {
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(cudaFree(base_ptr));
}

CUDAMemoryPool* CUDAMemoryPool::create(size_t size, unsigned gpu_id) {
	void* baseptr;
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(cudaMalloc(&baseptr, size));
	return new CUDAMemoryPool(static_cast<char*>(baseptr), size, gpu_id);
}

CUDAHostMemoryPool::CUDAHostMemoryPool(char* base_ptr, size_t size):
	MemoryPool(base_ptr, size) {}

CUDAHostMemoryPool::~CUDAHostMemoryPool() {
	CUDA_CALL(cudaFreeHost(base_ptr));
}

CUDAHostMemoryPool* CUDAHostMemoryPool::create(size_t size) {
	void* baseptr;
	CUDA_CALL(cudaHostAlloc(&baseptr, size, cudaHostAllocPortable));
	return new CUDAHostMemoryPool(static_cast<char*>(baseptr), size);
}

}
