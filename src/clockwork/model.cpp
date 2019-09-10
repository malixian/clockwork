#include "clockwork/model.h"
#include <dmlc/io.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include <fstream>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>

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

tvm::runtime::NDArray* readParams(const std::string &data) {
  dmlc::MemoryStringStream stream(const_cast<std::string*>(&data));
  tvm::runtime::NDArray* a = new tvm::runtime::NDArray();
  a->Load(&stream);
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
	readFileAsString(cold->params, params);
}

WarmModel* CoolModel::load() {
	return new WarmModel(this);
}

WarmModel::WarmModel(CoolModel* cool) {
	readMinModel(cool->clockwork, clockwork);
	warm = new so::TVMWarmSharedObject(cool->so.filename);
	params = readParams(cool->params);
	// TODO: don't treat params as NDarray, it's unnecessary
}

}