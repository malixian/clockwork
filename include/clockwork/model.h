#ifndef _CLOCKWORK_MODEL_H_
#define _CLOCKWORK_MODEL_H_

#include <string>

namespace clockwork{
namespace model {

class HotModel {
public:
	virtual int inputsize() = 0;
	virtual int outputsize() = 0;
	virtual void setinput(void* ptr) = 0;
	virtual void getoutput(void* ptr) = 0;

	virtual void unload() = 0;
	virtual void call() = 0;
};

class WarmModel {
public:
	virtual int size() = 0;
	virtual HotModel* load(void* ptr) = 0;
	virtual void unload() = 0;
};

class CoolModel {
public:
	virtual WarmModel* load() = 0;
	virtual void unload() = 0;
};

class ColdModel {
public:
	virtual CoolModel* load() = 0;
	virtual void unload() = 0;
};

extern ColdModel* FromDisk(std::string so, std::string clockwork, std::string params);

}
}

#endif