#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "clockwork/runtime.h"
#include "clockwork/clockwork.h"
#include <sstream>
#include <atomic>
#include <thread>
#include <fstream>
#include <istream>
#include "clockwork/modeldata.h"
#include "clockwork/serializedmodel.h"
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>
#include "clockwork/tvm/decoupled_graph_runtime.h"

using namespace clockwork;

struct Test {
    std::vector<uint32_t> blahs;        // this is default value

    PODS_SERIALIZABLE(
        1, //version             
        PODS_MDR(blahs)) //mandatory
        // PODS_OPT(port))         // this field is optional
};

void dopods() {
	Test v1;
	v1.blahs.push_back(5);
	v1.blahs.push_back(10);
	v1.blahs.push_back(15);


    pods::ResizableOutputBuffer out;
    pods::BinarySerializer<decltype(out)> serializer(out);
    if (serializer.save(v1) != pods::Error::NoError)
    {
        std::cerr << "serialization error\n";
        return;
    }

    Test v2;

    pods::InputBuffer in(out.data(), out.size());
    pods::BinaryDeserializer<decltype(in)> deserializer(in);
    if (deserializer.load(v2) != pods::Error::NoError)
    {
        std::cerr << "deserialization error\n";
        return;
    }

    for (unsigned i = 0; i < v1.blahs.size(); i++) {
    	std::cout << v1.blahs[i] << std::endl;
    }
    std::cout << std::endl;

    for (unsigned i = 0; i < v2.blahs.size(); i++) {
    	std::cout << v2.blahs[i] << std::endl;
    }
}

void loadmodel() {



	const int dtype_code = kDLFloat;
	const int dtype_bits = 32;
	const int dtype_lanes = 1;
	const int device_type = kDLGPU;
	const int device_id = 0;

	std::string model = "/home/jcmace/modelzoo/resnet50/tesla-m40_batchsize1/tvm-model";


    clockwork::binary::MinModel m;

    std::ifstream infile;
    infile.open(model + ".clockwork");

    pods::InputStream in(infile);
    pods::BinaryDeserializer<decltype(in)> deserializer(in);
    if (deserializer.load(m) != pods::Error::NoError)
    {
        std::cerr << "deserialization error\n";
        return;
    }
    infile.close();
    std::cout << "loaded clockwork model" << std::endl;

    clockwork::binary::Test::testModel(m);

    // tvm::runtime::PackedFunc run = mod.GetFunction("run");
    // run();

}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

	loadmodel();

	std::cout << "end" << std::endl;
}
