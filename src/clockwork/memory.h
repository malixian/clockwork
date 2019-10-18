#ifndef _CLOCKWORK_MEMORY_H_
#define _CLOCKWORK_MEMORY_H_

#include <atomic>
#include <memory>
#include <unordered_map>
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "tbb/concurrent_queue.h"

namespace clockwork {

class RuntimeModel {
public:
	model::Model* model;
	std::atomic_flag in_use;
	int version;
	std::shared_ptr<Allocation> weights;

	RuntimeModel(model::Model* model);

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
	~MemoryManager();
};

class CUDAPageCache : public PageCache {
public:
	CUDAPageCache(char* baseptr, uint64_t total_size, uint64_t page_size, const bool allowEvictions);
	~CUDAPageCache();
};

IOCache* make_IO_cache();
PageCache* make_GPU_cache(size_t cache_size, size_t page_size);

}

#endif