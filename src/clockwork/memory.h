#ifndef _CLOCKWORK_MEMORY_H_
#define _CLOCKWORK_MEMORY_H_

#include <atomic>
#include <memory>
#include <unordered_map>
#include <deque>
#include <memory>
#include "clockwork/cache.h"
#include "clockwork/model/batched.h"
#include "tbb/concurrent_queue.h"

namespace clockwork {

class RuntimeModel {
public:
	model::BatchedModel* model;
	std::atomic_flag in_use;
	int version;
	std::shared_ptr<Allocation> weights;

	RuntimeModel(model::BatchedModel* model);

	bool try_lock();
	void lock();
	void unlock();

};

class ModelStore {
private:
	std::atomic_flag in_use;
	std::unordered_map<int, RuntimeModel*> models;

public:

	ModelStore();

	// This will delete all models that are in the modelstore
	~ModelStore();

	RuntimeModel* get(int model_id);
	bool contains(int model_id);
	void put(int model_id, RuntimeModel* model);
	bool put_if_absent(int model_id, RuntimeModel* model);

};

// Simpler than a page cache for now since paging isn't really necessary for small inputs
// Will allocate memory itself using cudaMallocHost
class IOCache {
private:
	char* baseptr;
	tbb::concurrent_queue<char*> ptrs;

public:
	const size_t page_size;

	IOCache(size_t total_size, size_t page_size);
	~IOCache();

	bool take(char* &ptr);
	void release(char* ptr);

};

class MemoryManager {
public:
	IOCache* io_cache;
	PageCache* weights_cache;
	PageCache* workspace_cache;
	ModelStore* models;

	MemoryManager(PageCache* weights_cache, PageCache* workspace_cache);

	// This will delete the caches and the model store, possibly triggering freeing of cache memory
	~MemoryManager();
};

class MemoryAllocation {
public:
	std::atomic_bool freed;
	char* ptr;
	size_t offset, size;

	MemoryAllocation(char* base_ptr, size_t offset, size_t size) : freed(false), ptr(base_ptr + offset), offset(offset), size(size) {}
};

// Simple manager for workspace memory that allocates in a circular buffer
class MemoryPool {
private:
	std::mutex mutex;

	// Currently outstanding allocations
	std::deque<std::shared_ptr<MemoryAllocation>> allocations;
	
	// The memory that we're managing
	char* base_ptr;
	size_t size;

public:

	MemoryPool(char* base_ptr, size_t size);

	// Allocate `amount` of memory; returns nullptr if out of memory
	std::shared_ptr<MemoryAllocation> alloc(size_t amount);

	// Return the memory back to the pool
	void free(std::shared_ptr<MemoryAllocation> &allocation);

};

IOCache* make_IO_cache();

}

#endif