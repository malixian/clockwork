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

namespace clockwork {
namespace model {

ColdModel* FromDisk(std::string so, std::string clockwork, std::string params) {
	return new ColdDiskModelImpl(so, clockwork, params);
}

OpExec::OpExec(OpDef &op, BackendPackedCFunc f) : op(op), f(f), workspace_offsets(op.workspace_allocs) {
    size = op.inputs.size();
    offsets.resize(size);
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

      offsets[i] = op.inputs[i].offset;
      
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

void OpExec::call(void* baseptr) {
	for (unsigned i = 0; i < size; i++) {
	  input_tensors[i]->data = static_cast<char*>(baseptr) + offsets[i];
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

	std::vector<void*> workspace_ptrs(workspace_offsets.size());
	for (unsigned i = 0; i < workspace_offsets.size(); i++) {
	  workspace_ptrs[i] = static_cast<char*>(baseptr) + workspace_offsets[i];
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

ModelExec::ModelExec(ModelDef &mm, clockwork::so::TVMWarmSharedObject* warm) : mm(mm), ops(mm.ops.size()), size(mm.total_memory) {
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

int ModelExec::inputsize() {
	return mm.inputs[0].size;
}

int ModelExec::outputsize() {
	return mm.outputs[0].size;
}

void ModelExec::setinput(void* baseptr, void* inputptr) {	
	void* dstptr = static_cast<char*>(baseptr) + mm.inputs[0].offset;
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

void ModelExec::getoutput(void* baseptr, void* outputptr) {
	void* srcptr = static_cast<char*>(baseptr) + mm.outputs[0].offset;
	cudaStream_t stream = tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream;
	CUDA_CALL(
		cudaMemcpyAsync(
			srcptr,
			outputptr, 
			mm.outputs[0].size, 
			cudaMemcpyDeviceToHost,
			stream
		)
	)

}

void ModelExec::call(void* baseptr) {
	for (unsigned i = 0; i < ops.size(); i++) {
	  // std::cout << "Op " << i << " has ";
	  ops[i]->call(baseptr);
	}
}

void readFileAsString(const std::string &filename, std::string &dst) {
	std::ifstream in(filename, std::ios::binary);
	dst = std::string(
    	std::istreambuf_iterator<char>(in), 
    	std::istreambuf_iterator<char>());
	in.close();
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
	readFileAsString(cold->clockwork, clockwork);

	std::string params;
	readFileAsString(cold->params, params);

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
	ModelDef::ReadFrom(cool->clockwork, clockwork_spec);
	this->clockwork = new ModelExec(clockwork_spec, so);

	// Don't do anything with params yet
	params = cool->params;
	paramsSize = cool->paramsSize;
}

WarmModelImpl::~WarmModelImpl() {
	delete so;
	delete this->clockwork;
}

int WarmModelImpl::size() {
	return clockwork->size;
}

HotModel* WarmModelImpl::load(void* ptr) {
	return new HotModelImpl(this, ptr);
}

void WarmModelImpl::unload() {
	delete this;
}

HotModelImpl::HotModelImpl(WarmModelImpl* warm, void* params) : params(params), clockwork(warm->clockwork) {
	// Do the CUDA memcpy
	cudaStream_t stream = tvm::runtime::ManagedCUDAThreadEntry::ThreadLocal()->stream;
	CUDA_CALL(
		cudaMemcpyAsync(
			params, 
			warm->params, 
			warm->paramsSize, 
			cudaMemcpyHostToDevice,
			stream
		)
	)

	so = warm->so->load();  // Loads CUDA code into memory
}

HotModelImpl::~HotModelImpl() {
	so->unload();
}

int HotModelImpl::inputsize() {
	return clockwork->inputsize();
}

int HotModelImpl::outputsize() {
	return clockwork->outputsize();
}

void HotModelImpl::setinput(void* ptr) {
	clockwork->setinput(params, ptr);
}

void HotModelImpl::getoutput(void* ptr) {
	clockwork->getoutput(params, ptr);
}

void HotModelImpl::call() {
	clockwork->call(params);
}

void HotModelImpl::unload() {
	delete this;
}


}
}