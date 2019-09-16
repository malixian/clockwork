#ifndef _CLOCKWORK_MODEL_H_
#define _CLOCKWORK_MODEL_H_

#include <string>
#include <vector>

namespace clockwork{
namespace model {

// A model that can be executed
class ExecModel {
public:
	virtual int inputsize() = 0;
	virtual int outputsize() = 0;
	virtual void setinput(void* ptr) = 0;
	virtual void getoutput(void* ptr) = 0;

	virtual void unload() = 0;
	virtual void call() = 0;
};

// A model that has loaded its GPU code and its params into GPU memory, and is nearly executable
// To be executable, all that remains is to give the hotmodel the temporary GPU workspace
// memory that it needs for intermediate calculations
class HotModel {
public:
	virtual int num_workspace_pages(int pagesize) = 0;
	virtual ExecModel* load(std::vector<char*> &workspace_pages) = 0;
	virtual void unload() = 0;
};

// A model that has been deserialized and instantiated into memory, but not to GPU yet
class WarmModel {
public:
	virtual int inputsize() = 0;
	virtual int outputsize() = 0;
	virtual int num_params_pages(int pagesize) = 0;
	virtual int num_workspace_pages(int pagesize) = 0;
	virtual HotModel* load(std::vector<char*> &params_pages) = 0;
	virtual void unload() = 0;
};

// A model whose binary data is loaded into memory but not deserialized or instantiated
class CoolModel {
public:
	virtual WarmModel* load() = 0;
	virtual void unload() = 0;
};

// A model that is completely unloaded, e.g. just the names of files
class ColdModel {
public:
	virtual CoolModel* load() = 0;
	virtual void unload() = 0;
};

extern ColdModel* FromDisk(std::string so, std::string clockwork, std::string params);
extern CoolModel* FromMemory(const void *so, size_t so_size, const void *cw,
    size_t cw_size, const void *params, size_t params_size);

}
}

#endif