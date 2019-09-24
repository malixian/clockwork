#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "clockwork/runtime.h"
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
#include "clockwork-convert/tvm_model.h"
#include "clockwork-convert/tvm_abstract_model.h"

using namespace clockwork;

/** Converts a TVM model into a lighterweight Clockwork model 

Clockwork models use the original .so object for the TVM model,
but a different params file, and a more efficient alternative to the json.

*/

void convert(int pagesize, std::string model_so, std::string model_json, std::string model_params, std::string clockwork_meta_out, std::string clockwork_params_out) {

	tvm_model::Model model = tvm_model::Model::LoadFromFile(model_json);
	tvm_model::Params params = tvm_model::Params::LoadFromFile(model_params);
	tvm_model::Allocs allocs = tvm_model::Allocs::ProfileModel(model_so, model_json, model_params);


	clockwork_model::Model model2 = clockwork_model::Model::fromTVM(model, params, allocs);

	clockwork::model::PageMappedModelDef pagemappedmodel;
	char* weights;
	int weightsSize;
	clockwork_model::makeModelDef(model2, pagesize, pagemappedmodel, weights, weightsSize);

    std::ofstream outfile;
    outfile.open(clockwork_meta_out);

    pods::OutputStream out(outfile);
    pods::BinarySerializer<decltype(out)> serializer(out);
    if (serializer.save(pagemappedmodel) != pods::Error::NoError)
    {
        std::cerr << "serialization error\n";
    } else {
    	std::cout << "serialize success\n";
    }

    outfile.close();


	std::ofstream params_out(clockwork_params_out, std::ofstream::binary);
	params_out.write(weights, weightsSize);
	params_out.close();
}

void show_usage() {
	std::cout << "Provide the name of a model, to convert it" << std::endl;
	std::cout << "Specify page size with -p flag" << std::endl;
}

int main(int argc, char *argv[]) {
	std::vector<std::string> non_argument_strings;

	int pagesize = 16 * 1024 * 1024;
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if ((arg == "-h") || (arg == "--help")) {
		    show_usage();
		    return 0;
		} else if ((arg == "-p") || (arg == "--pagesize")) {
		    pagesize = atoi(argv[++i]);
		} else {
		  non_argument_strings.push_back(arg);
		}
	}

	if (non_argument_strings.size() < 1) {
		std::cerr << "Expected input model filename, none given." << std::endl 
		          << "Execute with --help for usage information." << std::endl;
		return 1;
	}


	// tvm_model::Params params = tvm_model::Params::LoadFromFile(non_argument_strings[0] + ".params");
	// tvm_model::Params params2 = tvm_model::Params::LoadFromFile(non_argument_strings[1] + ".params");

	// std::unordered_map<std::string, std::string> datamap;
	// for (auto &p : params.data) {
	// 	std::string data(static_cast<char*>(p.second->dataptr()), p.second->Size());
	// 	std::string name = p.first;
	// 	datamap[data] = name;
	// }
	// int found = 0;
	// int unfound = 0;
	// for (auto &p : params2.data) {
	// 	std::string data(static_cast<char*>(p.second->dataptr()), p.second->Size());
	// 	if (datamap.find(data) == datamap.end()) {
	// 		std::cout << "Unable to find " << p.first << std::endl;
	// 		unfound++;
	// 	} else {
	// 		found++;
	// 		// std::cout << p.first << " is " << datamap[data] << std::endl;
	// 	}
	// }
	// std::cout << "found " << found << " unfound " << unfound << std::endl;

	std::string model = non_argument_strings[0];
	std::string so = model + ".so";
	std::string json = model + ".json";
	std::string params = model + ".params";
	std::string clockwork = model + ".clockwork";
	std::string clockwork_params = model + ".clockwork_params";

	std::cout << "Processing " << model << std::endl;
	std::cout << "  pagesize=" << pagesize << std::endl;

	convert(pagesize, so, json, params, clockwork, clockwork_params);

	return 0;
}
