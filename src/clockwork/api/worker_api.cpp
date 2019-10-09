#include "clockwork/util.h"
#include "clockwork/api/worker_api.h"
#include <sstream>

namespace clockwork {
namespace workerapi {

uint64_t initial_offset = util::now();

std::string millis(uint64_t t) {
	std::stringstream ss;
	ss << (t / 1000000) << "." << ((t % 1000000) / 100000); // Crude way of printing as ms
	return ss.str();	
}

std::string offset(uint64_t t) {
	if (t < initial_offset) t = initial_offset;
	return millis(t - initial_offset);
}

std::string window(uint64_t earliest, uint64_t latest) {
	uint64_t now = util::now();
	earliest = earliest < now ? 0 : (earliest - now);
	latest = latest < now ? 0 : (latest - now);
	std::stringstream ss;
	ss << "[" << millis(earliest) << ", " << millis(latest) << "]";
	return ss.str();		
}

std::string LoadModelFromDisk::str() {
	std::stringstream ss;
	ss << "A" << id << ":LoadModelFromDisk"
	   << " model=" << model_id
	   << " " << window(earliest, latest)
	   << " " << model_path;
	return ss.str();
}

std::string LoadWeights::str() {
	std::stringstream ss;
	ss << "A" << id << ":LoadWeights"
	   << " model=" << model_id
	   // << " gpu=" << gpu_id
	   << " " << window(earliest, latest);
	return ss.str();
}

std::string EvictWeights::str() {
	std::stringstream ss;
	ss << "A" << id << ":EvictWeights"
	   << " model=" << model_id
	   // << " gpu=" << gpu_id
	   << " " << window(earliest, latest);
	return ss.str();
}

std::string Infer::str() {
	std::stringstream ss;
	ss << "A" << id << ":Infer"
	   << " model=" << model_id
	   // << " gpu=" << gpu_id
	   // << " batch=" << gpu_id
	   << " " << window(earliest, latest);
	return ss.str();
}

std::string ErrorResult::str() {
	std::stringstream ss;
	ss << "R" << id << ":Error " << status << ": " << message;
	return ss.str();
}

std::string Timing::str() {
	std::stringstream ss;
	ss << "[" << offset(begin) << ", " << offset(end) <<"] (" << millis(duration) << ")";
	return ss.str();
}

std::string LoadModelFromDiskResult::str() {
	std::stringstream ss;
	ss << "R" << id << ":LoadModelFromDisk"
	   << " input=" << input_size
	   << " output=" << output_size
	   << " batch=[";
	for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
		if (i > 0) ss << ", ";
		ss << supported_batch_sizes[i];
	}
	ss << "] weights=" << weights_size_in_cache
	   << " duration=" << millis(duration);
	return ss.str();
}

std::string LoadWeightsResult::str() {
	std::stringstream ss;
	ss << "R" << id << ":LoadWeights"
	   << " duration=" << millis(duration);
	return ss.str();
}

std::string EvictWeightsResult::str() {
	std::stringstream ss;
	ss << "R" << id << ":EvictWeights"
	   << " duration=" << millis(duration);
	return ss.str();
}

std::string InferResult::str() {
	std::stringstream ss;
	ss << "R" << id << ":Infer"
	   << " exec=" << millis(exec.duration)
	   << " input=" << millis(copy_input.duration)
	   << " output=" << millis(copy_output.duration);
	return ss.str();
}

}
}