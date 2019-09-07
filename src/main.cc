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
	const tvm::runtime::PackedFunc load_module(*tvm::runtime::Registry::Get("module.loadfile_so"));
	tvm::runtime::Module mod_syslib = load_module(model + ".so", "so");

	// Graph structure
	std::ifstream json_in(model + ".json", std::ios::in);  // read as text
	std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
	json_in.close();

	// Construct TVM runtime
	std::shared_ptr<tvm::runtime::DecoupledGraphRuntime> rt = DecoupledGraphRuntimeCreateDirect(json_data, mod_syslib, device_type, device_id);
	tvm::runtime::Module mod = tvm::runtime::Module(rt);
	// const tvm::runtime::PackedFunc create_graph_runtime(*tvm::runtime::Registry::Get("tvm.decoupled_graph_runtime.create_contiguous"));
	// tvm::runtime::Module mod = create_graph_runtime(json_data, mod_syslib, device_type, device_id);
	
	// tvm::runtime::Module mod = ClockworkGraphRuntimeCreate(json_data, mod_syslib, device_type, device_id);


    // Read params from file
    std::ifstream params_in(model + ".params", std::ios::binary);  // read as binary
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

	  // // Pull out params blob
	  // tvm::runtime::PackedFunc get_const_params = mod.GetFunction("get_const_params");
	  // tvm::runtime::PackedFunc set_const_params = mod.GetFunction("set_const_params");
	  // tvm::runtime::NDArray const_params = get_const_params();

    // load the model onto device
    tvm::runtime::PackedFunc load_to_device = mod.GetFunction("load_to_device");
    load_to_device();


    tvm::runtime::PackedFunc extract_model = mod.GetFunction("extract_model_spec");
    clockwork::binary::MinModel* minmodel = static_cast<clockwork::binary::MinModel*>((void*) extract_model());


    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();

}

void modeldata() {

	std::ifstream file("main", std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::cout << "create reader size " << size << std::endl;

	binary::ModelCodeReader reader = clockwork::binary::ModelCodeReader::create(size);
	while (!reader.readFrom(file)) {};

	std::cout << "done" << std::endl;
}

void runtime () {
	Runtime* runtime;
	// runtime = newFIFOThreadpoolRuntime(4);
	// runtime = newGreedyRuntime(1, 4);
	runtime = newDecoupledRuntime();

	int expected = 0;
	std::atomic_int* actual = new std::atomic_int{0};
	for (unsigned requestID = 1; requestID < 6; requestID++) {
		RequestBuilder* b = runtime->newRequest();
		for (unsigned taskID = 0; taskID < requestID; taskID++) {
			expected++;
			TaskType type = TaskTypes[taskID%TaskTypes.size()];
			b = b->addTask(type, [=] {
				std::stringstream ss;
				ss << std::this_thread::get_id() << "  type-" << type << "   request-" << requestID << "   task-" << taskID << std::endl;
				std::cout << ss.str();
				actual->fetch_add(1);
			});
		}
		b->submit();
	}



	while (actual->load() < expected) {}
	std::cout << "shutting down" << std::endl;

	runtime->shutdown(true);
	delete actual;
}

int main(int argc, char *argv[]) {
	std::cout << "begin" << std::endl;

	//dopods();
		loadmodel();
	if (false) {
		runtime();
		modeldata();
	}

	std::cout << "end" << std::endl;
}
