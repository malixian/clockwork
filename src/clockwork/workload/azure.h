#ifndef _CLOCKWORK_WORKLOAD_AZURE_H_
#define _CLOCKWORK_WORKLOAD_AZURE_H_

#include <queue>
#include <cstdint>
#include <functional>
#include <vector>
#include "clockwork/client.h"
#include "tbb/concurrent_queue.h"
#include <random>
#include <sstream>
#include <string>
#include <fstream>
#include "dmlc/logging.h"

namespace azure {

std::string get_trace_dir() {
  auto tracedir = std::getenv("AZURE_TRACE_DIR");
  if (tracedir == nullptr) { return ""; }
  return tracedir == nullptr ? "" : std::string(tracedir);	
}

std::string get_trace(int workload_id = 1) {
  std::string tracedir = get_trace_dir();
  if (tracedir == "") {
    std::cerr << "AZURE_TRACE_DIR variable not set, exiting" << std::endl;
    exit(1);
  }

  if (workload_id < 1 || workload_id > 14) {
  	std::cerr << "Azure workload_id must be between 1 and 14 inclusive.  Got " << workload_id << std::endl;
    exit(1);
  }

  std::stringstream s;
  s << tracedir << "/invocations_per_function_md.svls.anon.d";
  if (workload_id < 10) s << "0";
  s << workload_id << ".csv";

  return s.str();
}

std::vector<std::string> split(std::string line) {
	std::vector<std::string> result;
	std::stringstream s(line);
	while (s.good()) {
		std::string substr;
		std::getline(s, substr, ',');
		result.push_back(substr);
	}
	return result;
}

std::vector<unsigned> process_trace_line(std::string line, unsigned start_index) {
	std::vector<std::string> splits = split(line);
	CHECK(splits.size() == (1440 + start_index)) << "Unexpected format for azure trace line " << line;
	std::vector<unsigned> result;
	for (unsigned i = start_index; i < splits.size(); i++) {
		result.push_back(atoi(splits[i].c_str()));
	}
	return result;
}

std::vector<std::vector<unsigned>> read_trace_data(std::string filename) {
	std::ifstream f(filename);

	std::vector<std::vector<unsigned>> results;

	std::string line;
	std::getline(f, line); // Skip headers
	while (std::getline(f, line)) {
		results.push_back(process_trace_line(line, 4));
	}

	return results;
}

std::vector<std::vector<unsigned>> load_trace(int workload_id = 1) {
	return read_trace_data(get_trace(workload_id));
}

}

#endif 

