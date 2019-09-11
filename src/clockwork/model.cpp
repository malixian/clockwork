#include "clockwork/model.h"
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

void readFileAsString(const std::string &filename, std::string &dst) {
	std::ifstream in(filename, std::ios::binary);
	dst = std::string(
    	std::istreambuf_iterator<char>(in), 
    	std::istreambuf_iterator<char>());
	in.close();
}

void readMinModel(const std::string &data, binary::MinModel &model) {
	pods::InputBuffer in(data.data(), data.size());
    pods::BinaryDeserializer<decltype(in)> deserializer(in);
    pods::Error status = deserializer.load(model);
    CHECK(status == pods::Error::NoError) << "Cannot deserialize minmodel";
}

ColdDiskModel::ColdDiskModel(
		std::string so, 
		std::string clockwork, 
		std::string params
	) : so(so), clockwork(clockwork), params(params) {
}

CoolModel* ColdDiskModel::load() {
	return new CoolModel(this);
}

CoolModel::CoolModel(ColdDiskModel* cold) :
	so(Memfile::readFrom(cold->so)) {
	readFileAsString(cold->clockwork, clockwork);

	std::string params;
	readFileAsString(cold->params, params);

	// malloc cuda pinned memory
	paramsSize = params.size();
	CUDA_CALL(cudaMallocHost(&this->params, paramsSize));
	std::memcpy(this->params, params.data(), paramsSize);
}

CoolModel::~CoolModel() {
	CUDA_CALL(cudaFreeHost(this->params));
	// TODO: delete so memfile
}

WarmModel* CoolModel::load() {
	return new WarmModel(this);
}

void CoolModel::unload() {
	delete this;
}

WarmModel::WarmModel(CoolModel* cool) {
	// Load shared object
	so = new so::TVMWarmSharedObject(cool->so.filename);

	// Deserialize minmodel data structure
	readMinModel(cool->clockwork, clockwork_spec);
	this->clockwork = new binary::WarmModel(clockwork_spec, so);

	// Don't do anything with params yet
	params = cool->params;
	paramsSize = cool->paramsSize;
}

WarmModel::~WarmModel() {
	delete so;
	delete this->clockwork;
}

int WarmModel::size() {
	return clockwork->size;
}

HotModel* WarmModel::load(void* ptr) {
	return new HotModel(this, ptr);
}

void WarmModel::unload() {
	delete this;
}

HotModel::HotModel(WarmModel* warm, void* params) : params(params), clockwork(warm->clockwork) {
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

HotModel::~HotModel() {
	so->unload();
}

void HotModel::call() {
	clockwork->call(params);
}

void HotModel::unload() {
	delete this;
}

}