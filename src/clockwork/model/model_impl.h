#ifndef _CLOCKWORK_MODEL_IMPL_H_
#define _CLOCKWORK_MODEL_IMPL_H_

#include <string>
#include <dmlc/io.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include "clockwork/model.h"
#include "clockwork/modeldef.h"
#include "clockwork/model/memfile.h"
#include "clockwork/model/so.h"

namespace clockwork{
namespace model {


// TVM Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args, int* type_codes, int num_args);



/** Implementation of TVM op */
class OpExec {
public:
	OpDef &op;

	std::vector<uint64_t> workspace_offsets;
  
	std::vector<uint64_t> offsets;
	std::vector<DLTensor*> input_tensors;
	std::vector<TVMValue> op_inputs;
	std::vector<int> op_tcodes;
	int size;

	BackendPackedCFunc f;

	OpExec(OpDef &op, BackendPackedCFunc f);
	~OpExec();

	void call(void* baseptr);

};

class ModelExec {
public:
	ModelDef &mm;
	const int size;
	std::vector<OpExec*> ops;

	ModelExec(ModelDef &mm, clockwork::so::TVMWarmSharedObject* warm);
	~ModelExec();

	int inputsize();
	int outputsize();
	void setinput(void* baseptr, void* ptr);
	void getoutput(void* baseptr, void* ptr);

	void call(void* baseptr);
};

class ColdDiskModelImpl : public ColdModel {
public:
	const std::string so, clockwork, params; // filenames
	ColdDiskModelImpl(std::string so, std::string clockwork, std::string params);
	CoolModel* load();
	void unload() {};
};

/** A model that's in-memory but not yet deserialized */
class CoolModelImpl : public CoolModel {
public:
	const Memfile so;
	std::string clockwork;
	void* params; // cuda pinned host memory
	int paramsSize;

	CoolModelImpl(ColdDiskModelImpl* cold);
	~CoolModelImpl();

	WarmModel* load();

	void unload();

};

/** A model that's been deserialized but isn't yet loaded to device */

// TODO: pin params in memory as void*
class WarmModelImpl : public WarmModel {
public:
	model::ModelDef clockwork_spec;
	ModelExec* clockwork;
	so::TVMWarmSharedObject* so;
	void* params;
	int paramsSize;
	

	WarmModelImpl(CoolModelImpl* cool);
	~WarmModelImpl();

	int size();
	HotModel* load(void* ptr);

	void unload();

};

/** A model that's ready to be inferenced */
class HotModelImpl : public HotModel {
public:
	ModelExec* clockwork;
	void* params;
	so::TVMHotSharedObject* so;

	HotModelImpl(WarmModelImpl* warm, void* params);
	~HotModelImpl();

	int inputsize();
	int outputsize();
	void setinput(void* ptr);
	void getoutput(void* ptr);

	void call();
	void unload();
};

}
}

#endif