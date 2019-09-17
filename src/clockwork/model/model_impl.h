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
#include <cuda_runtime.h>

namespace clockwork{
namespace model {


// TVM Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args, int* type_codes, int num_args);



/** Implementation of TVM op */
class OpExec {
public:
	PageMappedOpDef &op;
  
	std::vector<DLTensor*> input_tensors;
	std::vector<TVMValue> op_inputs;
	std::vector<int> op_tcodes;
	int size;

	BackendPackedCFunc f;

	OpExec(PageMappedOpDef &op, BackendPackedCFunc f);
	~OpExec();

	void call(std::vector<char*> &pages);

};

class ModelExec {
public:
	PageMappedModelDef &mm;
	std::vector<OpExec*> ops;
	std::array<cudaEvent_t, 2> events;

	ModelExec(PageMappedModelDef &mm, clockwork::so::TVMWarmSharedObject* warm);
	~ModelExec();

	int num_params_pages(int pagesize);
	int num_exec_pages(int pagesize);

	int inputsize();
	int outputsize();
	void setinput(std::vector<char*> &pages, void* ptr);
	void getoutput(std::vector<char*> &pages, void* ptr);

	void call(std::vector<char*> &pages);
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
	char* params; // cuda pinned host memory
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
	model::PageMappedModelDef clockwork_spec;
	ModelExec* clockwork;
	so::TVMWarmSharedObject* so;
	so::TVMHotSharedObject* hotso;
	std::array<cudaEvent_t, 2> events;
	char* params;
	int paramsSize;
	

	WarmModelImpl(CoolModelImpl* cool);
	~WarmModelImpl();

	int inputsize();
	int outputsize();
	int num_workspace_pages(int pagesize);
	int num_params_pages(int pagesize);
	HotModel* load(std::vector<char*> &params_pages);
	void unload();

};

/** A model that's nearly ready to be inferenced */
class HotModelImpl : public HotModel {
public:
	ModelExec* clockwork;
	std::vector<char*> params_pages;
	so::TVMHotSharedObject* so;

	HotModelImpl(WarmModelImpl* warm, std::vector<char*> params_pages);
	~HotModelImpl();

	int num_workspace_pages(int pagesize);
	ExecModel* load(std::vector<char*> &workspace_pages);
	void unload();
};

/** A model that has been given its scratch workspace and can execute */
class ExecModelImpl : public ExecModel {
public:
	ModelExec* clockwork;
	std::vector<char*> pages;

	ExecModelImpl(HotModelImpl* hot, std::vector<char*> &workspace_pages);
	~ExecModelImpl();

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