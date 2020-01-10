#ifndef _CLOCKWORK_NETWORK_WORKER_API_H_
#define _CLOCKWORK_NETWORK_WORKER_API_H_

#include "clockwork/api/worker_api.h"
#include "clockwork/network/message.h"

namespace clockwork {
namespace network {

class error_result_tx : public msg_protobuf_tx<RES_ERROR, ErrorResultProto, workerapi::ErrorResult> {
public:
  virtual void set(workerapi::ErrorResult &result) {
  	msg.set_action_id(result.id);
  	msg.set_action_type(result.action_type);
  	msg.set_status(result.status);
  	msg.set_message(result.message);
  }
};

class error_result_rx : public msg_protobuf_rx<RES_ERROR, ErrorResultProto, workerapi::ErrorResult> {
public:
  virtual void get(workerapi::ErrorResult &result) {
  	result.id = msg.action_id();
  	result.action_type = msg.action_type();
  	result.status = msg.status();
  	result.message = msg.message();
  }
};

class load_model_from_disk_action_tx : public msg_protobuf_tx<ACT_LOAD_MODEL_FROM_DISK, LoadModelFromDiskActionProto, workerapi::LoadModelFromDisk> {
public:
  virtual void set(workerapi::LoadModelFromDisk &action) {
  	msg.set_action_id(action.id);
  	msg.set_model_id(action.model_id);
  	msg.set_model_path(action.model_path);
  	msg.set_earliest(action.earliest);
  	msg.set_latest(action.latest);
  }
};

class load_model_from_disk_action_rx : public msg_protobuf_rx<ACT_LOAD_MODEL_FROM_DISK, LoadModelFromDiskActionProto, workerapi::LoadModelFromDisk> {
public:
  virtual void get(workerapi::LoadModelFromDisk &action) {
  	action.id = msg.action_id();
  	action.action_type = workerapi::loadModelFromDiskAction;
  	action.model_id = msg.model_id();
  	action.model_path = msg.model_path();
  	action.earliest = msg.earliest();
  	action.latest = msg.latest();
  }
};

class load_model_from_disk_result_tx : public msg_protobuf_tx<RES_LOAD_MODEL_FROM_DISK, LoadModelFromDiskResultProto, workerapi::LoadModelFromDiskResult> {
public:
  virtual void set(workerapi::LoadModelFromDiskResult &result) {
  	msg.set_action_id(result.id);
  	msg.set_input_size(result.input_size);
  	msg.set_output_size(result.output_size);
  	for (unsigned batch_size : result.supported_batch_sizes) {
  		msg.add_supported_batch_sizes(batch_size);
  	}
  	msg.set_weights_size_in_cache(result.weights_size_in_cache);
  	msg.mutable_timing()->set_begin(result.begin);
  	msg.mutable_timing()->set_end(result.end);
  	msg.mutable_timing()->set_duration(result.duration);
  }
};

class load_model_from_disk_result_rx : public msg_protobuf_rx<RES_LOAD_MODEL_FROM_DISK, LoadModelFromDiskResultProto, workerapi::LoadModelFromDiskResult> {
public:
  virtual void get(workerapi::LoadModelFromDiskResult &result) {
  	result.id = msg.action_id();
  	result.action_type = workerapi::loadModelFromDiskAction;
  	result.status = actionSuccess;
  	result.begin = msg.timing().begin();
  	result.end = msg.timing().end();
  	result.duration = msg.timing().duration();
  	result.input_size = msg.input_size();
  	result.output_size = msg.output_size();
  	for (unsigned i = 0; i < msg.supported_batch_sizes_size(); i++) {
  		result.supported_batch_sizes.push_back(msg.supported_batch_sizes(i));
  	}
  	result.weights_size_in_cache = msg.weights_size_in_cache();
  }
};

class load_weights_action_tx : public msg_protobuf_tx<ACT_LOAD_WEIGHTS, LoadWeightsActionProto, workerapi::LoadWeights> {
public:
  virtual void set(workerapi::LoadWeights &action) {
  	msg.set_action_id(action.id);
  	msg.set_model_id(action.model_id);
  	msg.set_gpu_id(action.gpu_id);
  	msg.set_earliest(action.earliest);
  	msg.set_latest(action.latest);
  	msg.set_expected_duration(action.expected_duration);
  }
};

class load_weights_action_rx : public msg_protobuf_rx<ACT_LOAD_WEIGHTS, LoadWeightsActionProto, workerapi::LoadWeights> {
public:
  virtual void get(workerapi::LoadWeights &action) {
  	action.id = msg.action_id();
  	action.action_type = workerapi::loadWeightsAction;
  	action.model_id = msg.model_id();
  	action.gpu_id = msg.gpu_id();
  	action.earliest = msg.earliest();
  	action.latest = msg.latest();
  	action.expected_duration = msg.expected_duration();
  }
};

class load_weights_result_tx : public msg_protobuf_tx<RES_LOAD_WEIGHTS, LoadWeightsResultProto, workerapi::LoadWeightsResult> {
public:
  virtual void set(workerapi::LoadWeightsResult &result) {
  	msg.set_action_id(result.id);
  	msg.mutable_timing()->set_begin(result.begin);
  	msg.mutable_timing()->set_end(result.end);
  	msg.mutable_timing()->set_duration(result.duration);
  }
};

class load_weights_result_rx : public msg_protobuf_rx<RES_LOAD_WEIGHTS, LoadWeightsResultProto, workerapi::LoadWeightsResult> {
public:
  virtual void get(workerapi::LoadWeightsResult &result) {
  	result.id = msg.action_id();
  	result.action_type = workerapi::loadWeightsAction;
  	result.status = actionSuccess;
  	result.begin = msg.timing().begin();
  	result.end = msg.timing().end();
  	result.duration = msg.timing().duration();
  }
};

class evict_weights_action_tx : public msg_protobuf_tx<ACT_EVICT_WEIGHTS, EvictWeightsActionProto, workerapi::EvictWeights> {
public:
  virtual void set(workerapi::EvictWeights &action) {
  	msg.set_action_id(action.id);
  	msg.set_model_id(action.model_id);
  	msg.set_gpu_id(action.gpu_id);
  	msg.set_earliest(action.earliest);
  	msg.set_latest(action.latest);
  }
};

class evict_weights_action_rx : public msg_protobuf_rx<ACT_EVICT_WEIGHTS, EvictWeightsActionProto, workerapi::EvictWeights> {
public:
  virtual void get(workerapi::EvictWeights &action) {
  	action.id = msg.action_id();
  	action.action_type = workerapi::evictWeightsAction;
  	action.model_id = msg.model_id();
  	action.gpu_id = msg.gpu_id();
  	action.earliest = msg.earliest();
  	action.latest = msg.latest();
  }
};

class evict_weights_result_tx : public msg_protobuf_tx<RES_EVICT_WEIGHTS, EvictWeightsResultProto, workerapi::EvictWeightsResult> {
public:
  virtual void set(workerapi::EvictWeightsResult &result) {
  	msg.set_action_id(result.id);
  	msg.mutable_timing()->set_begin(result.begin);
  	msg.mutable_timing()->set_end(result.end);
  	msg.mutable_timing()->set_duration(result.duration);
  }
};

class evict_weights_result_rx : public msg_protobuf_rx<RES_EVICT_WEIGHTS, EvictWeightsResultProto, workerapi::EvictWeightsResult> {
public:
  virtual void get(workerapi::EvictWeightsResult &result) {
  	result.id = msg.action_id();
  	result.action_type = workerapi::evictWeightsAction;
  	result.status = actionSuccess;
  	result.begin = msg.timing().begin();
  	result.end = msg.timing().end();
  	result.duration = msg.timing().duration();
  }
};

class infer_action_tx : public msg_protobuf_tx_with_body<ACT_INFER, InferActionProto, workerapi::Infer> {
public:
  virtual void set(workerapi::Infer &action) {
  	msg.set_action_id(action.id);
  	msg.set_model_id(action.model_id);
  	msg.set_gpu_id(action.gpu_id);
  	msg.set_earliest(action.earliest);
  	msg.set_latest(action.latest);
  	msg.set_expected_duration(action.expected_duration);
  	msg.set_batch_size(action.batch_size);
  	body_len_ = action.input_size;
  	body_ = action.input;
  }
};

class infer_action_rx : public msg_protobuf_rx_with_body<ACT_INFER, InferActionProto, workerapi::Infer> {
public:
  virtual void get(workerapi::Infer &action) {
  	action.id = msg.action_id();
  	action.action_type = workerapi::inferAction;
  	action.model_id = msg.model_id();
  	action.gpu_id = msg.gpu_id();
  	action.earliest = msg.earliest();
  	action.latest = msg.latest();
  	action.expected_duration = msg.expected_duration();
  	action.batch_size = msg.batch_size();
  	action.input_size = body_len_;
  	action.input = static_cast<char*>(body_);
  }
};

class infer_result_tx : public msg_protobuf_tx_with_body<RES_INFER, InferResultProto, workerapi::InferResult> {
public:
  virtual void set(workerapi::InferResult &result) {
  	msg.set_action_id(result.id);
  	msg.mutable_copy_input_timing()->set_begin(result.copy_input.begin);
  	msg.mutable_copy_input_timing()->set_end(result.copy_input.end);
  	msg.mutable_copy_input_timing()->set_duration(result.copy_input.duration);
  	msg.mutable_exec_timing()->set_begin(result.exec.begin);
  	msg.mutable_exec_timing()->set_end(result.exec.end);
  	msg.mutable_exec_timing()->set_duration(result.exec.duration);
  	msg.mutable_copy_output_timing()->set_begin(result.copy_output.begin);
  	msg.mutable_copy_output_timing()->set_end(result.copy_output.end);
  	msg.mutable_copy_output_timing()->set_duration(result.copy_output.duration);
  	body_len_ = result.output_size;
  	body_ = result.output;
  }
};

class infer_result_rx : public msg_protobuf_rx_with_body<RES_INFER, InferResultProto, workerapi::InferResult> {
public:
  virtual void get(workerapi::InferResult &result) {
  	result.id = msg.action_id();
  	result.action_type = workerapi::inferAction;
  	result.status = actionSuccess;
  	result.copy_input.begin = msg.copy_input_timing().begin();
  	result.copy_input.end = msg.copy_input_timing().end();
  	result.copy_input.duration = msg.copy_input_timing().duration();
  	result.exec.begin = msg.exec_timing().begin();
  	result.exec.end = msg.exec_timing().end();
  	result.exec.duration = msg.exec_timing().duration();
  	result.copy_output.begin = msg.copy_output_timing().begin();
  	result.copy_output.end = msg.copy_output_timing().end();
  	result.copy_output.duration = msg.copy_output_timing().duration();
  	result.output_size = body_len_;
  	result.output = static_cast<char*>(body_);
  }
};

class clear_cache_action_tx : public msg_protobuf_tx_with_body<ACT_CLEAR_CACHE, ClearCacheActionProto, workerapi::ClearCache> {
public:
  virtual void set(workerapi::ClearCache &action) {
	msg.set_action_id(action.id);
  }
};

class clear_cache_action_rx : public msg_protobuf_rx_with_body<ACT_CLEAR_CACHE, ClearCacheActionProto, workerapi::ClearCache> {
public:
  virtual void get(workerapi::ClearCache &action) {
	action.id = msg.action_id();
	action.action_type = workerapi::clearCacheAction;
  }
};

class clear_cache_result_tx : public msg_protobuf_tx_with_body<RES_CLEAR_CACHE, ClearCacheResultProto, workerapi::ClearCacheResult>{
public:
  virtual void set(workerapi::ClearCacheResult &result) {
	msg.set_action_id(result.id);
  }
};

class clear_cache_result_rx : public msg_protobuf_rx_with_body<RES_CLEAR_CACHE, ClearCacheResultProto, workerapi::ClearCacheResult>{
public:
  virtual void get(workerapi::ClearCacheResult &result) {
	result.id = msg.action_id();
	result.action_type = workerapi::clearCacheAction;
  }
};

class get_worker_state_action_tx : public msg_protobuf_tx_with_body<ACT_GET_WORKER_STATE, GetWorkerStateActionProto, workerapi::GetWorkerState> {
public:
  virtual void set(workerapi::GetWorkerState &action) {
	msg.set_action_id(action.id);
  }
};

class get_worker_state_action_rx : public msg_protobuf_rx_with_body<ACT_GET_WORKER_STATE, GetWorkerStateActionProto, workerapi::GetWorkerState> {
public:
  virtual void get(workerapi::GetWorkerState &action) {
	action.id = msg.action_id();
	action.action_type = workerapi::getWorkerStateAction;
  }
};

class get_worker_state_result_tx : public msg_protobuf_tx_with_body<RES_GET_WORKER_STATE, GetWorkerStateResultProto, workerapi::GetWorkerStateResult>{
public:
  virtual void set(workerapi::GetWorkerStateResult &result) {
	msg.set_action_id(result.id);
	auto msg_info = msg.mutable_worker_memory_info();
	workerapi::WorkerMemoryInfo& result_info = result.worker_memory_info;
	msg_info->set_weights_cache_total(result_info.weights_cache_total);
	msg_info->set_weights_cache_remaining(result_info.weights_cache_remaining);
	msg_info->set_io_pool_total(result_info.io_pool_total);
	msg_info->set_io_pool_remaining(result_info.io_pool_remaining);
	msg_info->set_workspace_pool_total(result_info.workspace_pool_total);
	msg_info->set_workspace_pool_remaining(result_info.workspace_pool_remaining);
	for (unsigned i = 0; i < result_info.models.size(); i++) {
		ModelInfoProto* model = msg_info->add_models();
		model->set_id(result_info.models[i].id);
		model->set_size(result_info.models[i].size);
	}
  }
};

class get_worker_state_result_rx : public msg_protobuf_rx_with_body<RES_GET_WORKER_STATE, GetWorkerStateResultProto, workerapi::GetWorkerStateResult>{
public:
  virtual void get(workerapi::GetWorkerStateResult &result) {
	result.id = msg.action_id();
	result.action_type = workerapi::getWorkerStateAction;
	workerapi::WorkerMemoryInfo& result_info = result.worker_memory_info;
	auto msg_info = msg.worker_memory_info();
	result_info.weights_cache_total = msg_info.weights_cache_total();
	result_info.weights_cache_remaining = msg_info.weights_cache_remaining();
	result_info.io_pool_total = msg_info.io_pool_total();
	result_info.io_pool_remaining = msg_info.io_pool_remaining();
	result_info.workspace_pool_total = msg_info.workspace_pool_total();
	result_info.workspace_pool_remaining = msg_info.workspace_pool_remaining();
	for (unsigned i = 0; i < msg_info.models_size(); i++) {
		workerapi::ModelInfo model;
		model.id = msg_info.models(i).id();
		model.size = msg_info.models(i).size();
		result_info.models.push_back(model);
	}
  }
};

}
}

#endif
