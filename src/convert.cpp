#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "clockwork/runtime.h"
#include "clockwork/clockwork.h"
#include <sstream>
#include <atomic>
#include <thread>
#include <fstream>
#include <istream>
#include "clockwork/modeldef.h"
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

/** Converts a TVM model into a lighterweight Clockwork model 

Clockwork models use the original .so object for the TVM model,
but a different params file, and a more efficient alternative to the json.

*/

void convert(std::string model_so, std::string model_json, std::string model_params, std::string clockwork_meta_out, std::string clockwork_params_out) {

	const int dtype_code = kDLFloat;
	const int dtype_bits = 32;
	const int dtype_lanes = 1;
	const int device_type = kDLGPU;
	const int device_id = 0;

	const tvm::runtime::PackedFunc load_module(*tvm::runtime::Registry::Get("module.loadfile_so"));
	tvm::runtime::Module mod_syslib = load_module(model_so, "so");

	// Graph structure
	std::ifstream json_in(model_json, std::ios::in);  // read as text
	std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
	json_in.close();

	// Construct TVM runtime
	std::shared_ptr<tvm::runtime::DecoupledGraphRuntime> rt = DecoupledGraphRuntimeCreateDirect(json_data, mod_syslib, device_type, device_id);
	tvm::runtime::Module mod = tvm::runtime::Module(rt);
	// const tvm::runtime::PackedFunc create_graph_runtime(*tvm::runtime::Registry::Get("tvm.decoupled_graph_runtime.create_contiguous"));
	// tvm::runtime::Module mod = create_graph_runtime(json_data, mod_syslib, device_type, device_id);
	
	// tvm::runtime::Module mod = ClockworkGraphRuntimeCreate(json_data, mod_syslib, device_type, device_id);


    // Read params from file
    std::ifstream params_in(model_params, std::ios::binary);  // read as binary
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
    clockwork::model::ModelDef* minmodel = static_cast<clockwork::model::ModelDef*>((void*) extract_model());


    std::ofstream outfile;
    outfile.open(clockwork_meta_out);

    pods::OutputStream out(outfile);
    pods::BinarySerializer<decltype(out)> serializer(out);
    if (serializer.save(*minmodel) != pods::Error::NoError)
    {
        std::cerr << "serialization error\n";
    } else {
    	std::cout << "serialize success\n";
    }

    outfile.close();


	std::ofstream params_out(clockwork_params_out, std::ofstream::binary);

	tvm::runtime::PackedFunc get_const_params = mod.GetFunction("get_const_params");
	tvm::runtime::NDArray params = get_const_params();

	void* ptr = params.dataptr();
	uint64_t size = params.Size();

	params_out.write((const char*) ptr, size);
	params_out.close();

	// TODO: don't dump the ndarray, dump the actual bytes, 
	// bypassing NDarray class
}

void show_usage() {
	std::cout << "Provide the name of a model, to convert it" << std::endl;
}

int main(int argc, char *argv[]) {
	std::vector<std::string> non_argument_strings;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if ((arg == "-h") || (arg == "--help")) {
		    show_usage();
		    return 0;
		} else {
		  non_argument_strings.push_back(arg);
		}
	}

	if (non_argument_strings.size() < 1) {
		std::cerr << "Expected input model filename, none given." << std::endl 
		          << "Execute with --help for usage information." << std::endl;
		return 1;
	}

	std::string model = non_argument_strings[0];
	std::string so = model + ".so";
	std::string json = model + ".json";
	std::string params = model + ".params";
	std::string clockwork = model + ".clockwork";
	std::string clockwork_params = model + ".clockwork_params";

	convert(so, json, params, clockwork, clockwork_params);

	return 0;
}
