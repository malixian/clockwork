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
	std::string params;

	CoolModel(ColdDiskModel* cold);

	WarmModel* load();

	void unload() {
		// TODO: delete memfile
	}

};

/** A model that's been deserialized but isn't yet loaded to device */
class WarmModel {
public:
	binary::MinModel clockwork;
	so::TVMWarmSharedObject* warm;
	tvm::runtime::NDArray* params;

	WarmModel(CoolModel* cool);

};

/** A model that's ready to be inferenced */
class HotModel {

};

}

#endif