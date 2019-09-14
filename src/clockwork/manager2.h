#ifndef _CLOCKWORK_MANAGERR_H_
#define _CLOCKWORK_MANAGERR_H_

#include "clockwork/model.h"
#include "clockwork/memory.h"
#include "tvm/runtime/cuda_common.h"
#include <cuda_runtime.h>
#include <atomic>

namespace clockwork {

class Manager {
public:
	PageCache* cache;

	Manager(uint64_t cuda_prealloc_size, uint64_t cuda_page_size) {
		void* baseptr;
		CUDA_CALL(cudaMalloc(&baseptr, cuda_prealloc_size));
		cache = new PageCache(baseptr, cuda_prealloc_size, cuda_page_size);

	}


};

class ManagedModel;

class ParamsEvictionHandler : public EvictionHandler {
private:
	ManagedModel* model;
public:
	ParamsEvictionHandler(ManagedModel* model) : model(model) {}
	void evicted();
};

class WorkspaceEvictionHandler : public EvictionHandler {
private:
	ManagedModel* model;
public:
	WorkspaceEvictionHandler(ManagedModel* model) : model(model) {}
	void evicted();
};

enum ModelState { Cold, Cool, Warm, Hot, Exec };

class ManagedModel {
public:
	Manager* manager;

	model::ColdModel* cold = nullptr;
	model::CoolModel* cool = nullptr;
	model::WarmModel* warm = nullptr;
	model::HotModel* hot = nullptr;
	model::ExecModel* exec = nullptr;

	std::atomic_flag in_use = false;

	std::shared_ptr<Allocation> params_pages = nullptr;
	std::shared_ptr<Allocation> workspace_pages = nullptr;

	ParamsEvictionHandler* params_handler = nullptr;
	WorkspaceEvictionHandler* workspace_handler = nullptr;

private:

	ManagedModel(Manager* manager, ColdModel* cold) : manager(manager) {
		this->cold = cold;
		this->cool = this->cold->load();
		this->warm = this->cool->load();

		params_handler = new ParamsEvictionHandler(this);
		workspace_handler = new WorkspaceEvictionHandler(this);
	}

	void lock() {
		

		if (workspace_pages != nullptr) {

		}


	}

	void params_evicted() {
		if (exec != nullptr) {
			exec->unload();
			exec = nullptr;
		}
		if (hot != nullptr) {
			hot->unload();
		}
		manager->cache->free(workspace_pages);
		workspace_pages = nullptr;
		params_pages = nullptr;
	}

	void workspace_evicted() {
		if (exec != nullptr) {
			exec->unload();
			exec = nullptr;
		}
		workspace_pages = nullptr;
	}
};


class EvictionCallback {
public:
	virtual void evicted() = 0;
};

};

}

#endif