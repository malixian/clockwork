#include <iostream>
#include "tbb/task_scheduler_init.h"
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

struct TVM_Input {
	int batchsize;
	std::string name;
	std::string model_json_filename;
	std::string model_so_filename;
	std::string model_params_filename;
};

struct ConvertConfig {
	int weights_page_size;
	int workspace_page_size;
	std::vector<TVM_Input> inputs;
	std::string output_dir;
	std::string output_filename_prefix;
};

/** Converts a TVM model into a lighterweight Clockwork model 

Clockwork models use the original .so object for the TVM model,
but a different params file, and a more efficient alternative to the json.

*/

void convert(ConvertConfig config) {
	clockwork_model::PageMappedStorage* weights_mapping = nullptr;

	// The base output filename
	std::string outfile_base = config.output_dir + "/" + config.output_filename_prefix;

	std::cout << "Converting:" << std::endl;
	std::cout << "   weights_page_size=" << config.weights_page_size << std::endl;
	std::cout << "   workspace_page_size=" << config.workspace_page_size << std::endl;
	for (TVM_Input input : config.inputs) {
		std::cout << "  batch=" << input.batchsize << " " << input.model_params_filename << std::endl;
	}
	std::cout << "Outputting to " << outfile_base << std::endl;


	for (TVM_Input input : config.inputs) {

		std::cout << "Processing " << input.name << " batchsize=" << input.batchsize << std::endl;

		// Load the TVM stuff
		tvm_model::Model model = tvm_model::Model::LoadFromFile(input.model_json_filename);
		tvm_model::Params params = tvm_model::Params::LoadFromFile(input.model_params_filename);
		tvm_model::Allocs allocs = tvm_model::Allocs::ProfileModel(input.model_so_filename, input.model_json_filename, input.model_params_filename);

		// Convert into a clockwork model
		clockwork_model::Model model2 = clockwork_model::Model::fromTVM(model, params, allocs);

		// Map the paged storage
		clockwork_model::PageMappedStorage* mapped;
		clockwork::model::PageMappedModelDef pagemappedmodel;
		char* weights;
		int weightsSize;
		clockwork_model::makeModelDef(model2, config.weights_page_size, config.workspace_page_size, pagemappedmodel, weights, weightsSize, mapped, weights_mapping);

		// Save the model's metadata
		std::stringstream clockwork_meta_out;
		clockwork_meta_out << outfile_base << "." << input.batchsize << ".clockwork";

	    std::ofstream outfile;
	    outfile.open(clockwork_meta_out.str());

	    pods::OutputStream out(outfile);
	    pods::BinarySerializer<decltype(out)> serializer(out);
	    if (serializer.save(pagemappedmodel) != pods::Error::NoError)
	    {
	        std::cerr << "serialization error\n";
	    } else {
	    	std::cout << "serialize success\n";
	    }
	    outfile.close();

	    // Copy the SO
	    std::stringstream so_out;
	    so_out << outfile_base << "." << input.batchsize << ".so";

	    std::ifstream  src(input.model_so_filename, std::ios::binary);
	    std::ofstream  dst(so_out.str(), std::ios::binary);
		dst << src.rdbuf();

	    // First time round, save the weights
	    if (weights_mapping == nullptr) {
	    	std::string clockwork_params_out = outfile_base + ".clockwork_params";

			std::ofstream params_out(clockwork_params_out, std::ofstream::binary);
			params_out.write(weights, weightsSize);
			params_out.close();

			weights_mapping = mapped;
		}
	}
}

void show_usage() {
	std::cout << "Provide the name of a model, to convert it" << std::endl;
	std::cout << "Specify page size with -p flag" << std::endl;
}

int main(int argc, char *argv[]) {
	std::vector<std::string> non_argument_strings;

	ConvertConfig config;
	config.weights_page_size = 16 * 1024 * 1024;
	config.workspace_page_size = 64 * 1024 * 1024;
	config.output_dir = ".";
	config.output_filename_prefix = "model";

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if ((arg == "-h") || (arg == "--help")) {
		    show_usage();
		    return 0;
		} else if ((arg == "--weights_page_size")) {
		    config.weights_page_size = atoi(argv[++i]);
		} else if ((arg == "--workspace_page_size")) {
		    config.workspace_page_size = atoi(argv[++i]);
		} else if ((arg == "-o") || (arg == "--output")) {
		    config.output_dir = argv[++i];
		} else if ((arg == "-p") || (arg == "--page_size")) {
		    config.workspace_page_size = atoi(argv[++i]);
		    config.weights_page_size = config.workspace_page_size;
		} else {
		  non_argument_strings.push_back(arg);
		}
	}

	if (non_argument_strings.size() < 1) {
		std::cerr << "Expected input model filename, none given." << std::endl 
		          << "Execute with --help for usage information." << std::endl;
		return 1;
	}

	for (unsigned i = 1; i < non_argument_strings.size(); i+=2) {
		TVM_Input input;
		input.batchsize = atoi(non_argument_strings[i-1].c_str());
		input.name = non_argument_strings[i];
		input.model_json_filename = non_argument_strings[i] + ".json";
		input.model_so_filename = non_argument_strings[i] + ".so";
		input.model_params_filename = non_argument_strings[i] + ".params";

		config.inputs.push_back(input);
	}

	try {
		convert(config);
	} catch (std::string &s) {
		std::cout << "Convert failed: " << s << std::endl;
	}

	return 0;
}
