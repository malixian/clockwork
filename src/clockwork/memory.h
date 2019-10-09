#ifndef _CLOCKWORK_MEMORY_H_
#define _CLOCKWORK_MEMORY_H_

#include <atomic>
#include <memory>
#include <unordered_map>
#include "clockwork/cache.h"
#include "clockwork/model/model.h"

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

class MemoryManager {
public:
	PageCache* weights_cache;
	PageCache* workspace_cache;
	PageCache* io_cache;
	ModelStore* models;

	MemoryManager(PageCache* cache);
	MemoryManager(PageCache* weights_cache, PageCache* workspace_cache);
	~MemoryManager();
};

class CUDAPageCache : public PageCache {
public:
	CUDAPageCache(char* baseptr, uint64_t total_size, uint64_t page_size, const bool allowEvictions);
	~CUDAPageCache();
};

PageCache* make_IO_cache(size_t cache_size, size_t page_size);
PageCache* make_GPU_cache(size_t cache_size, size_t page_size = 16777216);

}

#endif