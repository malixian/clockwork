#include <dmlc/logging.h>
#include <tvm/runtime/cuda_common.h>
#include "clockwork/util.h"
#include "clockwork/model/model_impl2.h"

using namespace clockwork::model;

Model::Model(Memfile so_memfile, std::string &serialized_spec, int weights_size, char* weights_pinned_host_memory):
		so_memfile(so_memfile),	
		serialized_spec(serialized_spec), 
		weights_size(weights_size),
		weights_pinned_host_memory(weights_pinned_host_memory) {

	for (unsigned i = 0; i < rate_limit_events.size(); i++) {
		CUDA_CALL(cudaEventCreateWithFlags(&rate_limit_events[i], cudaEventDisableTiming));
	}
}

Model::~Model() {
	CUDA_CALL(cudaFreeHost(weights_pinned_host_memory));
}


char* Model::getpage(unsigned i) {
	if (i < weights_pages_count) return (*weights_pages)[i];
	if (i < total_pages_count) return (*workspace_pages)[i - weights_pages_count];
	return nullptr;
}

void Model::instantiate_model_on_host() {
	CHECK(warm_so == nullptr) << "instantiate_model_on_host warm_so is not nullptr";
	CHECK(spec == nullptr) << "instantiate_model_on_host spec is not nullptr";
	CHECK(op_execs == nullptr) << "instantiate_model_on_host op_execs is not nullptr";

	// 1: dlopen the TVM shared object and extract functions
	warm_so = new so::TVMWarmSharedObject(so_memfile.filename);

	// 2: deserialize the model metadata
	spec = new model::PageMappedModelDef();
	PageMappedModelDef::ReadFrom(serialized_spec, *spec);
	weights_pages_count = spec->weights_pages.size();
	workspace_pages_count = spec->total_pages - spec->weights_pages.size();
	total_pages_count = spec->total_pages;

	// 3: setup model executor
	op_execs = new std::vector<OpExec>(spec->ops.size());
	for (unsigned i = 0; i < spec->ops.size(); i++) {
		make_op_exec(spec->ops[i], (*op_execs)[i]);
	}
}

void Model::uninstantiate_model_on_host() {
	CHECK(warm_so != nullptr) << "uninstantiate_model_on_host warm_so is nullptr";
	CHECK(spec != nullptr) << "uninstantiate_model_on_host spec is nullptr";
	CHECK(op_execs != nullptr) << "uninstantiate_model_on_host op_execs is nullptr";
	delete warm_so;
	delete op_execs;
	delete spec;
	warm_so = nullptr;
	op_execs = nullptr;
	spec = nullptr;
}

void Model::instantiate_model_on_device() {
	CHECK(hot_so == nullptr) << "instantiate_model_on_device hot_so is not nullptr";

	/* 1: load the CUDA module onto device, which ultimately calls cuModuleLoad
	cuModuleLoad requires a barrier on kernel execution, and will block until
	current outstanding kernels have completed.  It will also block submission
	of any new kernels. */
	hot_so = warm_so->load();
}

void Model::uninstantiate_model_on_device() {
	CHECK(hot_so != nullptr) << "uninstantiate_model_on_device hot_so is nullptr";
	hot_so->unload();
	hot_so = nullptr;
}

unsigned Model::num_weights_pages(unsigned page_size) {
	CHECK(spec != nullptr) << "num_weights_pages spec is nullptr";
	CHECK(spec->configured_page_size == page_size)
			<< "Clockwork model was configured with mismatched page size, found "
			<< spec->configured_page_size << ", expected " << page_size;
	return weights_pages_count;
}

void Model::set_weights_pages(std::vector<char*> &weights_pages) {
	CHECK(this->weights_pages == nullptr) << "set_weights_pages current pages are not nullptr";
	this->weights_pages = new std::vector<char*>(weights_pages);
}

void Model::unset_weights_pages() {
	CHECK(this->weights_pages != nullptr) << "unset_weights_pages current pages are nullptr";
	delete this->weights_pages;
	this->weights_pages = nullptr;
}

unsigned Model::num_workspace_pages(unsigned page_size) {
	CHECK(spec != nullptr) << "num_workspace_pages spec is nullptr";
	CHECK(spec->configured_page_size == page_size)
			<< "Clockwork model was configured with mismatched page size, found "
			<< spec->configured_page_size << ", expected " << page_size;
	return workspace_pages_count;
}

void Model::set_workspace_pages(std::vector<char*> &workspace_pages) {
	CHECK(this->workspace_pages == nullptr) << "set_workspace_pages workspace_pages are not nullptr";
	this->workspace_pages = new std::vector<char*>(workspace_pages);
}

void Model::unset_workspace_pages() {
	CHECK(this->workspace_pages != nullptr) << "unset_workspace_pages workspace_pages are nullptr";
	delete this->workspace_pages;
	this->workspace_pages = nullptr;
}

void Model::transfer_weights_to_device(cudaStream_t stream) {
	CHECK(weights_pages != nullptr) << "transfer_weights_to_device weights_pages is nullptr";

	for (unsigned i = 0; i < weights_pages_count; i++) {
		PageDef &def = spec->weights_pages[i];
		CUDA_CALL(
			cudaMemcpyAsync(
				(*weights_pages)[i], // dstptr
				weights_pinned_host_memory + def.base_offset, // srcptr
				def.size, // size 
				cudaMemcpyHostToDevice,
				stream
			)
		)
		CUDA_CALL(cudaEventSynchronize(rate_limit_events[i % MAX_OUTSTANDING_EVENTS]));
		CUDA_CALL(cudaEventRecord(rate_limit_events[i % MAX_OUTSTANDING_EVENTS], stream));
	}
}

unsigned Model::input_size() {
	/** TODO: for now, a model only has one input */
	CHECK(spec != nullptr) << "input_size spec is nullptr";
	return spec->inputs[0].size;
}

/* Preconditions: instantiate_model_on_host && set_workspace_pages */
void Model::transfer_input_to_device(char* input_ptr, cudaStream_t stream) {
	CHECK(spec != nullptr) << "transfer_input_to_device spec is nullptr";
	CHECK(workspace_pages != nullptr) << "transfer_input_to_device workspace_pages are nullptr";
	void* dst_ptr = getpage(spec->inputs[0].page) + spec->inputs[0].page_offset;
	CUDA_CALL(
		cudaMemcpyAsync(
			dst_ptr,
			input_ptr, 
			spec->inputs[0].size,
			cudaMemcpyHostToDevice,
			stream
		)
	)
}

/* Preconditions: instantiate_model_on_host */
unsigned Model::output_size() {
	/** TODO: for now, a model only has one output */
	CHECK(spec != nullptr) << "output_size spec is nullptr";
	return spec->outputs[0].size;
}

/* Preconditions: instantiate_model_on_host && set_workspace_pages */
void Model::transfer_output_from_device(char* output_ptr, cudaStream_t stream) {
	CHECK(spec != nullptr) << "transfer_output_from_device spec is nullptr";
	CHECK(workspace_pages != nullptr) << "transfer_output_from_device workspace_pages are nullptr";
	void* src_ptr = getpage(spec->outputs[0].page) + spec->outputs[0].page_offset;
	CUDA_CALL(
		cudaMemcpyAsync(
			output_ptr, 
			src_ptr,
			spec->outputs[0].size, 
			cudaMemcpyDeviceToHost,
			stream
		)
	)
}

/* Preconditions: instantiate_model_on_device && set_workspace_pages && set_weights_pages */
void Model::call(cudaStream_t stream) {
	CHECK(hot_so != nullptr) << "call hot_so is nullptr";
	CHECK(weights_pages != nullptr) << "call weights_pages is nullptr";
	CHECK(workspace_pages != nullptr) << "call workspace_pages is nullptr";

	clockwork::util::SetStream(stream);

	for (unsigned i = 0; i < op_execs->size(); i++) {
		call_op_exec((*op_execs)[i]);
		CUDA_CALL(cudaEventSynchronize(rate_limit_events[i % MAX_OUTSTANDING_EVENTS]));
		CUDA_CALL(cudaEventRecord(rate_limit_events[i % MAX_OUTSTANDING_EVENTS], stream));
	}
}

void Model::make_op_exec(PageMappedOpDef &spec, OpExec &op) {
	op.spec = &spec;
	
	op.num_inputs = spec.inputs.size();

	op.input_tensors.resize(op.num_inputs);
	op.func_inputs.resize(op.num_inputs);
	op.func_tcodes.resize(op.num_inputs);

	for (unsigned i = 0; i < op.num_inputs; i++) {
		auto &tensor = op.input_tensors[i];
		tensor.data = nullptr;
		tensor.ctx = DLContext{kDLGPU, 0}; // TODO: multiple devices
		tensor.ndim = spec.inputs[i].shape.size();
		tensor.dtype = DLDataType{kDLFloat, 32, 1};
		tensor.shape = spec.inputs[i].shape.data();
		tensor.strides = nullptr;
		tensor.byte_offset = 0;
		op.func_inputs[i].v_handle = &tensor;
		op.func_tcodes[i] = kArrayHandle;
	}

	op.workspace_ptrs.resize(spec.workspace_allocs.size());

	op.so_function_name = this->spec->so_functions[spec.so_function];
	op.f = reinterpret_cast<OpFunc>(warm_so->so.GetSymbol(op.so_function_name.c_str()));
}

void Model::call_op_exec(OpExec &op) {
	// Point the inputs to the right place
	for (unsigned i = 0; i < op.num_inputs; i++) {
		auto &tensor = op.input_tensors[i];
		auto &spec = op.spec->inputs[i];
		tensor.data = getpage(spec.page) + spec.page_offset;
	}
	// Set the workspace alloc pointers
	for (unsigned i = 0; i < op.workspace_ptrs.size(); i++) {
		auto &spec = op.spec->workspace_allocs[i];
		op.workspace_ptrs[i] = getpage(spec.page) + spec.page_offset;
	}
	clockwork::so::TVMBackendWorkspaceManager::Set(op.workspace_ptrs);

	int ret = (*(op.f))(
	  op.func_inputs.data(),
	  op.func_tcodes.data(), 
	  op.num_inputs
	);
	clockwork::so::TVMBackendWorkspaceManager::Clear();
	CHECK_EQ(ret, 0) << TVMGetLastError();
}

Model* Model::loadFromDisk(
		std::string so_filename, 
		std::string clockwork_filename,
		std::string clockwork_weights_filename ) {

	Memfile so_memfile = Memfile::readFrom(so_filename);

	std::string clockwork_serialized_spec;
	util::readFileAsString(clockwork_filename, clockwork_serialized_spec);

	std::string weights;
	util::readFileAsString(clockwork_weights_filename, weights);
	int weights_size = weights.size();
	char* weights_pinned_host_memory;
	CUDA_CALL(cudaMallocHost(&weights_pinned_host_memory, weights_size));
	std::memcpy(weights_pinned_host_memory, weights.data(), weights_size);

	return new Model(
		so_memfile, 
		clockwork_serialized_spec, 
		weights_size, 
		weights_pinned_host_memory );
}