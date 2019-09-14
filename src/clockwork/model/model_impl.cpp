#include "clockwork/model.h"
#include "clockwork/model/model_impl.h"
#include <dmlc/io.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include <fstream>
#include <tvm/runtime/cuda_common.h>
#include <cuda_runtime.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <cstring>
#include "clockwork/util.h"

namespace clockwork {
namespace model {

ColdModel* FromDisk(std::string so, std::string clockwork, std::string params) {
	return new ColdDiskModelImpl(so, clockwork, params);
}

OpExec::OpExec(PageMappedOpDef &op, BackendPackedCFunc f) : op(op), f(f) {
    size = op.inputs.size();
    op_inputs.resize(size);
    op_tcodes.resize(size);
    input_tensors.resize(size);

    for (unsigned i = 0; i < size; i++) {
      DLTensor* input = new DLTensor();
      input->data = nullptr;
      input->ctx = DLContext{kDLGPU, 0};
      input->ndim = op.inputs[i].shape.size();
      input->dtype = DLDataType{kDLFloat, 32, 1};
      input->shape = op.inputs[i].shape.data();
      input->strides = nullptr;
      input->byte_offset = 0;
      
      input_tensors[i] = input;
      op_inputs[i].v_handle = input;
      op_tcodes[i] = kArrayHandle;
    }
}

OpExec::~OpExec() {
    for (unsigned i = 0; i < size; i++) {
      delete input_tensors[i];
    }
}

void OpExec::call(std::vector<char*> &pages) {
	for (unsigned i = 0; i < size; i++) {
		input_tensors[i]->data = pages[op.inputs[i].page] + op.inputs[i].page_offset;
	}

	// std::cout << size << " inputs" << std::endl;
	// for (unsigned i = 0; i < op_inputs.size(); i++) {
	//   DLTensor* tensor = static_cast<DLTensor*>(op_inputs[i].v_handle);
	//   std::cout << "Input " << i << " ndim=" << tensor->ndim << "shape=[";
	//   for (unsigned j = 0; j < tensor->ndim; j++) {
	//     std::cout << *(static_cast<int64_t*>(tensor->shape) + j) << " ";
	//   }
	//   std::cout << "]" << " datatype=" << tensor->dtype.code << "-" << tensor->dtype.bits << "-" << tensor->dtype.lanes << " stridesnull=" << (tensor->strides==nullptr) << " offset=" << tensor->byte_offset << std::endl;
	// }

	std::vector<void*> workspace_ptrs(op.workspace_allocs.size());
	for (unsigned i = 0; i < op.workspace_allocs.size(); i++) {
		workspace_ptrs[i] = pages[op.workspace_allocs[i].page] + op.workspace_allocs[i].page_offset;
	}
	clockwork::so::TVMBackendWorkspaceManager::Set(workspace_ptrs);

	int ret = (*f)(
	  op_inputs.data(),
	  op_tcodes.data(), 
	  size
	);
	clockwork::so::TVMBackendWorkspaceManager::Clear();
	CHECK_EQ(ret, 0) << TVMGetLastError();
}

ModelExec::ModelExec(PageMappedModelDef &mm, clockwork::so::TVMWarmSharedObject* warm) : mm(mm), ops(mm.ops.size()) {
	// Extract the SO functions
	std::vector<BackendPackedCFunc> fs(mm.so_functions.size());
	for (unsigned i = 0; i < mm.so_functions.size(); i++) {
		fs[i] = reinterpret_cast<BackendPackedCFunc>(warm->so.GetSymbol(mm.so_functions[i].c_str()));
	}

	for (unsigned i = 0; i < mm.ops.size(); i++) {
		ops[i] = new OpExec(mm.ops[i], fs[mm.ops[i].so_function]);

		// TODO: no need at the moment, but later, eager load cuda functions
	}

	// TODO: should be able to handle more than one input and output
	CHECK(mm.inputs.size() == 1) << "Expected model to have 1 input, but found " << mm.inputs.size();
	CHECK(mm.outputs.size() == 1) << "Expected model to have 1 input, but found " << mm.inputs.size();
}

ModelExec::~ModelExec() {
	for (unsigned i = 0; i < ops.size(); i++) {
		delete ops[i];
	}
}

int ModelExec::num_params_pages(int pagesize) {
	CHECK(pagesize == mm.configured_page_size) 
		<< "Clockwork model was configured with wrong page size, found " 
		<< mm.configured_page_size 
		<< ", expected " 
		<< pagesize;
	return mm.weights_pages.size();
}

int ModelExec::num_exec_pages(int pagesize) {
	CHECK(pagesize == mm.configured_page_size) 
		<< "Clockwork model was configured with wrong page size, found " 
		<< mm.configured_page_size 
		<< ", expected " 
		<< pagesize;
	return mm.total_pages - mm.weights_pages.size();
}

int ModelExec::inputsize() {
	return mm.inputs[0].size;
}

int ModelExec::outputsize() {
	return mm.outputs[0].size;
}

void ModelExec::setinput(std::vector<char*> &pages, void* inputptr) {
	void* dstptr = pages[mm.inputs[0].page] + mm.inputs[0].page_offset;
	cudaStream_t stream = tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream;
	CUDA_CALL(
		cudaMemcpyAsync(
			dstptr,
			inputptr, 
			mm.inputs[0].size, 
			cudaMemcpyHostToDevice,
			stream
		)
	)
}

void ModelExec::getoutput(std::vector<char*> &pages, void* outputptr) {
	void* srcptr = pages[mm.outputs[0].page] + mm.outputs[0].page_offset;
	cudaStream_t stream = tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream;
	CUDA_CALL(
		cudaMemcpyAsync(
			outputptr, 
			srcptr,
			mm.outputs[0].size, 
			cudaMemcpyDeviceToHost,
			stream
		)
	)

}

void ModelExec::call(std::vector<char*> &pages) {
	for (unsigned i = 0; i < ops.size(); i++) {
	  ops[i]->call(pages);
	}
}

ColdDiskModelImpl::ColdDiskModelImpl(
		std::string so, 
		std::string clockwork, 
		std::string params
	) : so(so), clockwork(clockwork), params(params) {
}

CoolModel* ColdDiskModelImpl::load() {
	return new CoolModelImpl(this);
}

CoolModelImpl::CoolModelImpl(ColdDiskModelImpl* cold) :
	so(Memfile::readFrom(cold->so)) {
	clockwork::util::readFileAsString(cold->clockwork, clockwork);

	std::string params;
	clockwork::util::readFileAsString(cold->params, params);

	// malloc cuda pinned memory
	paramsSize = params.size();
	CUDA_CALL(cudaMallocHost(&this->params, paramsSize));
	std::memcpy(this->params, params.data(), paramsSize);
}

CoolModelImpl::~CoolModelImpl() {
	CUDA_CALL(cudaFreeHost(this->params));
	// TODO: delete so memfile
}

WarmModel* CoolModelImpl::load() {
	return new WarmModelImpl(this);
}

void CoolModelImpl::unload() {
	delete this;
}

WarmModelImpl::WarmModelImpl(CoolModelImpl* cool) {
	// Load shared object
	so = new so::TVMWarmSharedObject(cool->so.filename);

	// Deserialize minmodel data structure
	PageMappedModelDef::ReadFrom(cool->clockwork, this->clockwork_spec);
	this->clockwork = new ModelExec(clockwork_spec, so);

	// Don't do anything with params yet
	params = cool->params;
	paramsSize = cool->paramsSize;
}

WarmModelImpl::~WarmModelImpl() {
	delete so;
	delete this->clockwork;
}

int WarmModelImpl::inputsize() {
	return clockwork->inputsize();
}

int WarmModelImpl::outputsize() {
	return clockwork->outputsize();
}

int WarmModelImpl::num_workspace_pages(int pagesize) {
	return clockwork->num_exec_pages(pagesize);
}

int WarmModelImpl::num_params_pages(int pagesize) {
	return clockwork->num_params_pages(pagesize);
}

HotModel* WarmModelImpl::load(std::vector<char*> &params_pages) {
	return new HotModelImpl(this, params_pages);
}

void WarmModelImpl::unload() {
	delete this;
}

HotModelImpl::HotModelImpl(WarmModelImpl* warm, std::vector<char*> params_pages) : params_pages(params_pages), clockwork(warm->clockwork) {
	// Do the CUDA memcpy
	cudaStream_t stream = tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream;

	for (unsigned i = 0; i < warm->clockwork_spec.weights_pages.size(); i++) {
		PageDef &def = warm->clockwork_spec.weights_pages[i];
		CUDA_CALL(
			cudaMemcpyAsync(
				params_pages[i],                // dstptr
				warm->params + def.base_offset, // srcptr
				def.size,                       // size 
				cudaMemcpyHostToDevice,
				stream
			)
		)
	}

	so = warm->so->load();  // Loads CUDA code into memory
}

HotModelImpl::~HotModelImpl() {
	so->unload();
}

int HotModelImpl::num_workspace_pages(int pagesize) {
	return clockwork->num_exec_pages(pagesize);
}

ExecModel* HotModelImpl::load(std::vector<char*> &workspace_pages) {
	return new ExecModelImpl(this, workspace_pages);
}

void HotModelImpl::unload() {
	delete this;
}

ExecModelImpl::ExecModelImpl(HotModelImpl* hot, std::vector<char*> &workspace_pages) : clockwork(hot->clockwork) {
	pages.reserve(hot->params_pages.size() + workspace_pages.size());
	for (unsigned i = 0; i < hot->params_pages.size(); i++) {
		pages.push_back(hot->params_pages[i]);
	}
	for (unsigned i = 0; i < workspace_pages.size(); i++) {
		pages.push_back(workspace_pages[i]);
	}
}

ExecModelImpl::~ExecModelImpl() {}

int ExecModelImpl::inputsize() {
	return clockwork->inputsize();
}

int ExecModelImpl::outputsize() {
	return clockwork->outputsize();
}

void ExecModelImpl::setinput(void* ptr) {
	clockwork->setinput(pages, ptr);
}

void ExecModelImpl::getoutput(void* ptr) {
	clockwork->getoutput(pages, ptr);
}

void ExecModelImpl::call() {
	clockwork->call(pages);
}

void ExecModelImpl::unload() {
	delete this;
}


}
}