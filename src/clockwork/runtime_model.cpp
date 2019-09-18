#include <dmlc/logging.h>
#include "clockwork/runtime_model.h"

using namespace clockwork;

RuntimeModel::RuntimeModel(PageCache* cache, model::ColdModel* cold) : cache(cache), cold(cold) {
}

RuntimeModel::State RuntimeModel::lock() {
	if (!cache->trylock(params_alloc)) return State::Warm;
	if (!cache->trylock(workspace_alloc)) return State::Hot;
	return State::Exec;
}

void RuntimeModel::ensureState(State state) {
	// TODO: don't do this here
	if (state == State::Exec) return;
	if (exec != nullptr) execToHot();

	if (state == State::Hot) return;
	if (hot != nullptr) hotToWarm();
}

void RuntimeModel::unlock() {
	cache->unlock(workspace_alloc);
	cache->unlock(params_alloc);
}

int RuntimeModel::inputsize() {
	return warm->inputsize();
}

int RuntimeModel::outputsize() {
	return warm->outputsize();
}

void RuntimeModel::evict() {
	cache->free(workspace_alloc); // Triggers transition exec -> hot if workspace_alloc is valid
	cache->free(params_alloc); // Triggers transition hot -> warm if params_alloc is valid
}

void RuntimeModel::coldToCool() {
	CHECK(cold != nullptr) << "Cannot transition cold -> cool, cold == nullptr";
	CHECK(cool == nullptr) << "Cannot transition cold -> cool, cool already exists";
	cool = cold->load();
}

void RuntimeModel::coolToWarm() {
	CHECK(cool != nullptr) << "Cannot transition cool -> warm, cool == nullptr";
	CHECK(warm == nullptr) << "Cannot transition cool -> warm, cool already exists";
	warm = cool->load();
}

void RuntimeModel::warmToHot() {
	CHECK(warm != nullptr) << "Cannot transition warm -> hot, warm == nullptr";
	CHECK(hot == nullptr) << "Cannot transition warm -> hot, hot != nullptr";
	CHECK(params_alloc == nullptr) << "Cannot transition warm -> hot, params_alloc already allocated";
	params_alloc = cache->alloc(warm->num_params_pages(cache->page_size), []{});
	std::vector<char*> page_ptrs(params_alloc->pages.size());
	for (unsigned i = 0; i < params_alloc->pages.size(); i++) {
		page_ptrs[i] = params_alloc->pages[i]->ptr;
	}
	hot = warm->load(page_ptrs);
}

void RuntimeModel::hotToExec() {
	CHECK(hot != nullptr) << "Cannot transition hot -> exec, hot == nullptr";
	CHECK(exec == nullptr) << "Cannot transition hot -> exec, exec != nullptr";
	CHECK(workspace_alloc == nullptr) << "Cannot transition hot -> exec, workspace_alloc already allocated";
	workspace_alloc = cache->alloc(hot->num_workspace_pages(cache->page_size), []{});
	std::vector<char*> page_ptrs(workspace_alloc->pages.size());
	for (unsigned i = 0; i < workspace_alloc->pages.size(); i++) {
		page_ptrs[i] = workspace_alloc->pages[i]->ptr;
	}
	exec = hot->load(page_ptrs);
}

void RuntimeModel::setInput(void* input) {
	CHECK(exec != nullptr) << "Cannot set input on exec == nullptr";
	exec->setinput(input);
}

void RuntimeModel::call() {
	CHECK(exec != nullptr) << "Cannot call exec == nullptr";
	exec->call();
}

void RuntimeModel::getOutput(void* output) {
	CHECK(exec != nullptr) << "Cannot get output of exec == nullptr";
	exec->getoutput(output);
}

void RuntimeModel::execToHot() {
	CHECK(exec != nullptr) << "Cannot transition exec -> hot, exec == nullptr";
	CHECK(workspace_alloc != nullptr) << "Cannot transition exec -> hot, workspace_alloc == nullptr";
	cache->free(workspace_alloc);
	workspace_alloc = nullptr;
	exec->unload();
	exec = nullptr;
}

void RuntimeModel::hotToWarm() {
	if (exec != nullptr) execToHot();
	CHECK(hot != nullptr) << "Cannot transition hot -> warm, hot == nullptr";
	CHECK(params_alloc != nullptr) << "Cannot transition hot -> warm, params_alloc == nullptr";
	cache->free(params_alloc);
	params_alloc = nullptr;
	hot->unload();
	hot = nullptr;
}

void RuntimeModel::warmToCool() {
	if (hot != nullptr) hotToWarm();
	CHECK(warm != nullptr) << "Cannot transition warm -> cool, warm == nullptr";
	warm->unload();
	warm = nullptr;
}

void RuntimeModel::coolToCold() {
	if (warm != nullptr) warmToCool();
	CHECK(cool != nullptr) << "Cannot transition cool -> cold, cool == nullptr";
	cool->unload();
	cool = nullptr;
}