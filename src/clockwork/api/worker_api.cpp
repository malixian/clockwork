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

std::string ClearCache::str() {
	std::stringstream ss;
	ss << "A" << id << ":ClearCache";
	return ss.str();
}

std::string GetWorkerState::str() {
	std::stringstream ss;
	ss << "A" << id << ":GetWorkerState";
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

std::string ModelInfo::str() {
	std::stringstream ss;
	ss << "M-" << id
	   << " src=" << source
	   << " input=" << input_size
	   << " output=" << output_size
	   << " batch=[";
	for (unsigned i = 0; i < supported_batch_sizes.size(); i++) {
		if (i > 0) ss << ", ";
		ss << supported_batch_sizes[i];
	}
	ss << "] weights=" << weights_size
	   << " (" << num_weights_pages << " pages)";
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

std::string ClearCacheResult::str() {
	std::stringstream ss;
	ss << "R" << id << ":ClearCache";
	return ss.str();
}

float as_gb(size_t size) {
	return size / ((float) (1024*1024*1024));
}
float as_mb(size_t size) {
	return size / ((float) (1024*1024));	
}

std::string GPUInfo::str() {
	std::stringstream ss;
	ss.precision(1);
	ss << std::fixed;
	ss << "GPU-" << id
	   << " weights_cache=" << as_gb(weights_cache_size) << "GB (" << weights_cache_total_pages << " pages)"
	   << " io_pool=" << as_mb(io_pool_size) << "MB"
	   << " workspace_pool=" << as_mb(workspace_pool_size) << "MB"
	   << " " << models.size() << " models currently on GPU";
	return ss.str();
}

std::string WorkerMemoryInfo::str() {
	std::stringstream ss;
	ss << " page_size=" << page_size << "\n"
	   << "gpus=\n";
	for (auto &gpu : gpus) {
		ss << " " << gpu.str() << "\n";
	}
	ss << "models=\n";
	for (auto &model : models) {
		ss << " " << model.str() << "\n";
	}
	return ss.str();
}

std::string GetWorkerStateResult::str() {
	std::stringstream ss;
	ss << "R" << id << ":GetWorkerState:\n";
	ss << worker.str();
	return ss.str();
}

}
}
