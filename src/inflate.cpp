#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include "clockwork/telemetry.h"
#include "clockwork/runtime.h"
#include <fstream>
#include <sstream>
#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <pods/streams.h>

std::unordered_map<std::string, std::string> processTask(clockwork::SerializedTaskTelemetry &t) {
	std::unordered_map<std::string, std::string> o;

	o["queue"] = std::to_string(t.dequeued - t.eligible_for_dequeue);
	o["cpu"] = std::to_string(t.exec_complete - t.dequeued);
	o["cuda"] = std::to_string(t.async_duration);
	o["hostside_e2e"] = std::to_string(t.async_complete - t.dequeued);

	return o;
}

std::unordered_map<std::string, std::string> process(clockwork::SerializedRequestTelemetry &t) {
	std::unordered_map<std::string, std::string> o;
	o["request_id"] = std::to_string(t.request_id);
	o["model_id"] = std::to_string(t.model_id);
	o["arrived"] = std::to_string(t.arrived);
	o["submitted"] = std::to_string(t.submitted);
	o["complete"] = std::to_string(t.complete);
	o["execution_latency"] = std::to_string(t.complete - t.submitted);
	o["total_latency"] = std::to_string(t.complete - t.arrived);

	for (unsigned i = 0; i < t.tasks.size(); i++) {
		clockwork::SerializedTaskTelemetry task = t.tasks[i];
		std::string task_type = clockwork::TaskTypeName(clockwork::TaskTypes[task.task_type]);

		std::unordered_map<std::string, std::string> task_data = processTask(task);

		for (auto e : task_data) {
			o[task_type + "_" + e.first] = e.second;
		}
	}

	return o;
}

/** Inflates output clockwork telemetry to TSV file */
void inflate(std::string input_filename, std::string output_filename) {
	std::ifstream infile;
	infile.open(input_filename);

    pods::InputStream in(infile);
    pods::BinaryDeserializer<decltype(in)> deserializer(in);

    uint64_t start_time = 0;

    clockwork::SerializedRequestTelemetry telemetry;
    int count = 0;

    std::vector<std::string> headers = {{
    	"request_id",
    	"model_id",
    	"arrived", 
    	"submitted",
    	"complete",
    	"execution_latency",
    	"total_latency",
    }};

    for (unsigned i = 0; i < clockwork::TaskTypes.size(); i++) {
		std::string task_type = clockwork::TaskTypeName(clockwork::TaskTypes[i]);
		headers.push_back(task_type + "_queue");
		headers.push_back(task_type + "_cpu");
		headers.push_back(task_type + "_cuda");
		headers.push_back(task_type + "_hostside_e2e");
    }

    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (deserializer.load(telemetry) == pods::Error::NoError) {	
		std::unordered_map<std::string, std::string> row = process(telemetry);
		rows.push_back(row);
    }
    std::cout << "Processed " << rows.size() << " records" << std::endl;

    std::ofstream outfile;
    outfile.open(output_filename);

    int i = 0;
	for (auto header : headers) {
		if (i++ > 0) {
			outfile << "\t";
		}
		outfile << header;
	}
	outfile << "\n";

    for (auto row : rows) {
    	i = 0;
    	for (auto header : headers) {
    		if (i++ > 0) outfile << "\t";
    		outfile << row[header];
    	}
    	outfile << "\n";
    }

    outfile.close();
}

void show_usage() {
	std::cout << "Inflates a binary format telemetry file into a TSV" << std::endl;
	std::cout << "Specify input filename and output filename" << std::endl;
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
		std::cerr << "Expected input telemetry filename, none given." << std::endl 
		          << "Execute with --help for usage information." << std::endl;
		return 1;
	}

	std::string input_filename = non_argument_strings[0];
	std::string output_filename = input_filename + ".tsv";
	if (non_argument_strings.size() >= 2) {
		output_filename = non_argument_strings[1];
	}

	std::cout << "Inflating " << input_filename << std::endl
	          << "       to " << output_filename << std::endl;

	inflate(input_filename, output_filename);

	return 0;
}
