#ifndef _CLOCKWORK_NETWORK_CLIENT_API_H_
#define _CLOCKWORK_NETWORK_CLIENT_API_H_

#include "clockwork/api/client_api.h"
#include "clockwork/network/message.h"

namespace clockwork {
namespace network {

void set_header(RequestHeader &request_header, RequestHeaderProto* proto) {
	proto->set_user_id(request_header.user_id);
	proto->set_user_request_id(request_header.user_request_id);
}

void set_header(ResponseHeader &response_header, ResponseHeaderProto* proto) {
	proto->set_user_request_id(response_header.user_request_id);
	proto->set_status(response_header.status);
	proto->set_message(response_header.message);
}

void get_header(RequestHeader &request_header, const RequestHeaderProto &proto) {
	request_header.user_id = proto.user_id();
	request_header.user_request_id = proto.user_request_id();
}

void get_header(ResponseHeader &response_header, const ResponseHeaderProto &proto) {
	response_header.user_request_id = proto.user_request_id();
	response_header.status = proto.status();
	response_header.message = proto.message();
}

class msg_inference_req_tx : public msg_protobuf_tx_with_body<REQ_INFERENCE, ModelInferenceRequest, clientapi::InferenceRequest> {
public:
  virtual void set(clientapi::InferenceRequest &request) {
  	set_header(request.header, msg.mutable_header());
  	msg.set_model_id(request.model_id);
  	msg.set_batch_size(request.batch_size);
  	body_len_ = request.input_size;
  	body_ = request.input;
  }
};

class msg_inference_req_rx : public msg_protobuf_rx_with_body<REQ_INFERENCE, ModelInferenceRequest, clientapi::InferenceRequest> {
public:
  virtual void get(clientapi::InferenceRequest &request) {
  	get_header(request.header, msg.header());
  	request.model_id = msg.model_id();
  	request.batch_size = msg.batch_size();
  	request.input_size = body_len_;
  	request.input = body_;
  }
};

class msg_inference_rsp_tx : public msg_protobuf_tx_with_body<RSP_INFERENCE, ModelInferenceResponse, clientapi::InferenceResponse> {
public:
  void set(clientapi::InferenceResponse &response) {
  	set_header(response.header, msg.mutable_header());
  	msg.set_model_id(response.model_id);
  	msg.set_batch_size(response.batch_size);
  	body_len_ = response.output_size;
  	body_ = response.output;
  }
};

class msg_inference_rsp_rx : public msg_protobuf_rx_with_body<RSP_INFERENCE, ModelInferenceResponse, clientapi::InferenceResponse> {
public:
  void get(clientapi::InferenceResponse &response) {
    get_header(response.header, msg.header());
    response.model_id = msg.model_id();
    response.batch_size = msg.batch_size();
    response.output_size = body_len_;
    response.output = body_;
  }
};

class msg_evict_req_tx : public msg_protobuf_tx<REQ_EVICT, EvictRequest, clientapi::EvictRequest> {
public:
  void set(clientapi::EvictRequest &request) {
    set_header(request.header, msg.mutable_header());
    msg.set_model_id(request.model_id);
  }
};

class msg_evict_req_rx : public msg_protobuf_rx<REQ_EVICT, EvictRequest, clientapi::EvictRequest> {
public:
  void get(clientapi::EvictRequest &request) {
    get_header(request.header, msg.header());
    request.model_id = msg.model_id();
  }
};

class msg_evict_rsp_tx : public msg_protobuf_tx<RSP_EVICT, EvictRequest, clientapi::EvictRequest> {
public:
  void set(clientapi::EvictRequest &request) {
    set_header(request.header, msg.mutable_header());
    msg.set_model_id(request.model_id);
  }
};

class msg_evict_rsp_rx : public msg_protobuf_rx<RSP_EVICT, EvictRequest, clientapi::EvictRequest> {
public:
  void get(clientapi::EvictRequest &request) {
    get_header(request.header, msg.header());
    request.model_id = msg.model_id();
  }
};

class msg_load_remote_model_req_tx : public msg_protobuf_tx<REQ_LOAD_REMOTE_MODEL, LoadModelFromDiskRequest, clientapi::LoadModelFromRemoteDiskRequest> {
public:  
  void set(clientapi::LoadModelFromRemoteDiskRequest &request) {
    set_header(request.header, msg.mutable_header());
    msg.set_remote_path(request.remote_path);
  }
};

class msg_load_remote_model_req_rx : public msg_protobuf_rx<REQ_LOAD_REMOTE_MODEL, LoadModelFromDiskRequest, clientapi::LoadModelFromRemoteDiskRequest> {
public:
  void get(clientapi::LoadModelFromRemoteDiskRequest &request) {
    get_header(request.header, msg.header());
    request.remote_path = msg.remote_path();
  }
};

class msg_load_remote_model_rsp_tx : public msg_protobuf_tx<RSP_LOAD_REMOTE_MODEL, LoadModelFromDiskResponse, clientapi::LoadModelFromRemoteDiskResponse> {
public:
  void set(clientapi::LoadModelFromRemoteDiskResponse &response) {
    set_header(response.header, msg.mutable_header());
    msg.set_model_id(response.model_id);
    msg.set_input_size(response.input_size);
    msg.set_output_size(response.output_size);
  }
};

class msg_load_remote_model_rsp_rx : public msg_protobuf_rx<RSP_LOAD_REMOTE_MODEL, LoadModelFromDiskResponse, clientapi::LoadModelFromRemoteDiskResponse> {
public:
  void get(clientapi::LoadModelFromRemoteDiskResponse &response) {
    get_header(response.header, msg.header());
    response.model_id = msg.model_id();
    response.input_size = msg.input_size();
    response.output_size = msg.output_size();
  }
};

}
}

#endif