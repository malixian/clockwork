#ifndef _CLOCKWORK_MODEL_H_
#define _CLOCKWORK_MODEL_H_

#include <string>
#include "clockwork/memfile.h"
#include <dmlc/io.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include "clockwork/serializedmodel.h"
#include "clockwork/so.h"

namespace clockwork{

class ColdModel;
class CoolModel;
class WarmModel;
class HotModel;

/** A model on disk; not really used */
class ColdModel {
public:
	virtual CoolModel* load() = 0;	
};

class ColdDiskModel : public ColdModel {
public:
	const std::string so, clockwork, params; // filenames
	
	ColdDiskModel(std::string so, std::string clockwork, std::string params);

	CoolModel* load();

};


/** A model that's in-memory but not yet deserialized */
class CoolModel {
public:
	const Memfile so;
	std::string clockwork;
	void* params; // cuda pinned host memory
	int paramsSize;

	CoolModel(ColdDiskModel* cold);
	~CoolModel();

	WarmModel* load();

	void unload();

};

/** A model that's been deserialized but isn't yet loaded to device */

// TODO: pin params in memory as void*
class WarmModel {
public:
	binary::MinModel clockwork_spec;
	binary::WarmModel* clockwork;
	so::TVMWarmSharedObject* so;
	void* params;
	int paramsSize;
	

	WarmModel(CoolModel* cool);
	~WarmModel();

	int size();
	HotModel* load(void* ptr);

	void unload();

};

/** A model that's ready to be inferenced */
class HotModel {
public:
	binary::WarmModel* clockwork;
	void* params;
	so::TVMHotSharedObject* so;

	HotModel(WarmModel* warm, void* params);
	~HotModel();

	void call();
	void unload();


};

}

#endif