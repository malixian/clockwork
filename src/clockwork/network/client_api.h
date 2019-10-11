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

class msg_inference_req_tx : public msg_protobuf_tx_with_body<REQ_INFERENCE, ModelInferenceReqProto, clientapi::InferenceRequest> {
public:
  virtual void set(clientapi::InferenceRequest &request) {
  	set_header(request.header, msg.mutable_header());
  	msg.set_model_id(request.model_id);
  	msg.set_batch_size(request.batch_size);
  	body_len_ = request.input_size;
  	body_ = request.input;
  }
};

class msg_inference_req_rx : public msg_protobuf_rx_with_body<REQ_INFERENCE, ModelInferenceReqProto, clientapi::InferenceRequest> {
public:
  virtual void get(clientapi::InferenceRequest &request) {
  	get_header(request.header, msg.header());
  	request.model_id = msg.model_id();
  	request.batch_size = msg.batch_size();
  	request.input_size = body_len_;
  	request.input = body_;
  }
};

class msg_inference_rsp_tx : public msg_protobuf_tx_with_body<RSP_INFERENCE, ModelInferenceRspProto, clientapi::InferenceResponse> {
public:
  void set(clientapi::InferenceResponse &response) {
  	set_header(response.header, msg.mutable_header());
  	msg.set_model_id(response.model_id);
  	msg.set_batch_size(response.batch_size);
  	body_len_ = response.output_size;
  	body_ = response.output;
  }
};

class msg_inference_rsp_rx : public msg_protobuf_rx_with_body<RSP_INFERENCE, ModelInferenceRspProto, clientapi::InferenceResponse> {
public:
  void get(clientapi::InferenceResponse &response) {
    get_header(response.header, msg.header());
    response.model_id = msg.model_id();
    response.batch_size = msg.batch_size();
    response.output_size = body_len_;
    response.output = body_;
  }
};

class msg_evict_req_tx : public msg_protobuf_tx<REQ_EVICT, EvictReqProto, clientapi::EvictRequest> {
public:
  void set(clientapi::EvictRequest &request) {
    set_header(request.header, msg.mutable_header());
    msg.set_model_id(request.model_id);
  }
};

class msg_evict_req_rx : public msg_protobuf_rx<REQ_EVICT, EvictReqProto, clientapi::EvictRequest> {
public:
  void get(clientapi::EvictRequest &request) {
    get_header(request.header, msg.header());
    request.model_id = msg.model_id();
  }
};

class msg_evict_rsp_tx : public msg_protobuf_tx<RSP_EVICT, EvictRspProto, clientapi::EvictResponse> {
public:
  void set(clientapi::EvictResponse &request) {
    set_header(request.header, msg.mutable_header());
  }
};

class msg_evict_rsp_rx : public msg_protobuf_rx<RSP_EVICT, EvictRspProto, clientapi::EvictResponse> {
public:
  void get(clientapi::EvictResponse &request) {
    get_header(request.header, msg.header());
  }
};

class msg_load_remote_model_req_tx : public msg_protobuf_tx<REQ_LOAD_REMOTE_MODEL, LoadModelFromDiskReqProto, clientapi::LoadModelFromRemoteDiskRequest> {
public:  
  void set(clientapi::LoadModelFromRemoteDiskRequest &request) {
    set_header(request.header, msg.mutable_header());
    msg.set_remote_path(request.remote_path);
  }
};

class msg_load_remote_model_req_rx : public msg_protobuf_rx<REQ_LOAD_REMOTE_MODEL, LoadModelFromDiskReqProto, clientapi::LoadModelFromRemoteDiskRequest> {
public:
  void get(clientapi::LoadModelFromRemoteDiskRequest &request) {
    get_header(request.header, msg.header());
    request.remote_path = msg.remote_path();
  }
};

class msg_load_remote_model_rsp_tx : public msg_protobuf_tx<RSP_LOAD_REMOTE_MODEL, LoadModelFromDiskRspProto, clientapi::LoadModelFromRemoteDiskResponse> {
public:
  void set(clientapi::LoadModelFromRemoteDiskResponse &response) {
    set_header(response.header, msg.mutable_header());
    msg.set_model_id(response.model_id);
    msg.set_input_size(response.input_size);
    msg.set_output_size(response.output_size);
  }
};

class msg_load_remote_model_rsp_rx : public msg_protobuf_rx<RSP_LOAD_REMOTE_MODEL, LoadModelFromDiskRspProto, clientapi::LoadModelFromRemoteDiskResponse> {
public:
  void get(clientapi::LoadModelFromRemoteDiskResponse &response) {
    get_header(response.header, msg.header());
    response.model_id = msg.model_id();
    response.input_size = msg.input_size();
    response.output_size = msg.output_size();
  }
};


class msg_upload_model_req_tx : public msg_protobuf_tx<REQ_UPLOAD_MODEL, ModelUploadReqProto, clientapi::UploadModelRequest> {
private:
  enum body_tx_state {
    BODY_SEND_SO,
    BODY_SEND_CLOCKWORK,
    BODY_SEND_CWPARAMS,
    BODY_SEND_DONE,
  } body_tx_state = BODY_SEND_SO;

  std::string blob_so, blob_cw, blob_cwparams;
public:
  void set(clientapi::UploadModelRequest &request) {
    set_header(request.header, msg.mutable_header());
    msg.set_params_len(request.weights_size);

    // For now assume one instance with a batch size of 1
    msg.set_so_len(request.instances[0].so_size);
    msg.set_clockwork_len(request.instances[0].spec_size);
  }

  virtual uint64_t get_tx_body_len() const {
    return blob_so.size() + blob_cw.size() + blob_cwparams.size();
  }

  virtual std::pair<const void *,size_t> next_tx_body_buf() {
    switch (body_tx_state) {
      case BODY_SEND_SO: {
        body_tx_state = BODY_SEND_CLOCKWORK;
        return std::make_pair(blob_so.data(), blob_so.size());
      }
      case BODY_SEND_CLOCKWORK: {
        body_tx_state = BODY_SEND_CWPARAMS;
        return std::make_pair(blob_cw.data(), blob_cw.size());        
      }
      case BODY_SEND_CWPARAMS: {
        body_tx_state = BODY_SEND_DONE;
        return std::make_pair(blob_cwparams.data(), blob_cwparams.size());
      }
      default: {
        CHECK(false) << "upload_model_req in invalid state";
      }
    }
  }
};

class msg_upload_model_req_rx : public msg_protobuf_rx<REQ_UPLOAD_MODEL, ModelUploadReqProto, clientapi::UploadModelRequest> {
private:
  enum body_rx_state {
    BODY_RX_SO,
    BODY_RX_CLOCKWORK,
    BODY_RX_CWPARAMS,
    BODY_RX_DONE,
  } body_rx_state = BODY_RX_SO;

  size_t body_len_ = 0;
  void* buf_so = nullptr;
  void* buf_clockwork = nullptr;
  void* buf_cwparams = nullptr;

public:
  void get(clientapi::UploadModelRequest &request) {
    get_header(request.header, msg.header());
    request.weights_size = msg.params_len();
    request.weights = buf_cwparams;
    
    clientapi::UploadModelRequest::ModelInstance instance;
    instance.batch_size = 1;
    instance.so_size = msg.so_len();
    instance.so = buf_so;
    instance.spec_size = msg.clockwork_len();
    instance.spec = buf_clockwork;
  }

  void set_body_len(size_t body_len) { body_len_ = body_len; }

  virtual void header_received(const void *hdr, size_t hdr_len) {
    msg_protobuf_rx::header_received(hdr, hdr_len);

    CHECK(msg.so_len() + msg.clockwork_len() + msg.params_len() == body_len_) 
      << "load_model body length did mot match expected length";

    buf_so = new uint8_t[msg.so_len()];
    buf_clockwork = new uint8_t[msg.clockwork_len()];
    buf_cwparams = new uint8_t[msg.params_len()];
  }

  virtual std::pair<void *,size_t> next_body_rx_buf() {
    switch (body_rx_state) {
      case BODY_RX_SO: {
        body_rx_state = BODY_RX_CLOCKWORK;
        return std::make_pair(buf_so, msg.so_len());        
      };
      case BODY_RX_CLOCKWORK: {
        body_rx_state = BODY_RX_CWPARAMS;
        return std::make_pair(buf_clockwork, msg.clockwork_len());
      };
      case BODY_RX_CWPARAMS: {
        body_rx_state = BODY_RX_DONE;
        return std::make_pair(buf_cwparams, msg.params_len());
      };
      default: CHECK(false) << "upload_model_req in invalid state";
    }
  }

  virtual void body_buf_received(size_t len)
  {
    size_t expected;

    if (body_rx_state == BODY_RX_CLOCKWORK) {
      expected = msg.so_len();
    } else if (body_rx_state == BODY_RX_CWPARAMS) {
      expected = msg.clockwork_len();
    } else if (body_rx_state == BODY_RX_DONE) {
      expected = msg.params_len();
    } else {
      throw "TODO";
    }

    if (expected != len)
      throw "unexpected body rx len";
  }
};

class msg_upload_model_rsp_tx : public msg_protobuf_tx<RSP_UPLOAD_MODEL, ModelUploadRspProto, clientapi::UploadModelResponse> {
public:
  void set(clientapi::UploadModelResponse &response) {
    set_header(response.header, msg.mutable_header());
    msg.set_model_id(response.model_id);
    msg.set_input_size(response.input_size);
    msg.set_output_size(response.output_size);
  }
};

class msg_upload_model_rsp_rx : public msg_protobuf_rx<RSP_UPLOAD_MODEL, ModelUploadRspProto, clientapi::UploadModelResponse> {
public:
  void get(clientapi::UploadModelResponse &response) {
    get_header(response.header, msg.header());
    response.model_id = msg.model_id();
    response.input_size = msg.input_size();
    response.output_size = msg.output_size();
  }
};

}
}

#endif