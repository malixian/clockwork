#ifndef _CLOCKWORK_RUNTIME_MODEL_H_
#define _CLOCKWORK_RUNTIME_MODEL_H_

#include <deque>
#include <unordered_map>
#include "clockwork/model.h"
#include "clockwork/cache.h"

/*

The runtime model hooks into the model implementation from the model subdiretory.
It handles state changes between the cold, cool, warm, hot, and exec.
It also maintains paged memory used by the model implementations.

*/

namespace clockwork {

/** Model is not concurrent, with the exception of the eviction handlers, which may be called
while holding the cache lock */
class RuntimeModel {
public:

	enum State { Warm, Hot, Exec };

private:

	PageCache* cache;

	model::ColdModel* cold;
	model::CoolModel* cool = nullptr;
	model::WarmModel* warm = nullptr;
	model::HotModel* hot = nullptr;
	model::ExecModel* exec = nullptr;

	std::shared_ptr<Allocation> params_alloc = nullptr;
	std::shared_ptr<Allocation> workspace_alloc = nullptr;

public:

	RuntimeModel(PageCache* cache, model::ColdModel* cold);

	State lock();
	void unlock();

	void ensureState(State state);

	void evict();

	int inputsize();
	int outputsize();

	void coldToCool();
	void coolToWarm();
	void warmToHot();
	void hotToExec();
	void setInput(void* input);
	void call();
	void getOutput(void* output);
	void execToHot();
	void hotToWarm();
	void warmToCool();
	void coolToCold();
	
};

}

#endif