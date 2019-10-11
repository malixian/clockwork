#ifndef CLOCKWORK_NET_CLOCKWORK_H_
#define CLOCKWORK_NET_CLOCKWORK_H_

#include <iostream>
#include <string>

#include "clockwork/network/client_api.h"
#include <clockwork/network.h>
#include <clockwork.pb.h>


template <class TMsg, uint64_t TMsgType>
class msg_protobuf_tx : public message_tx {
public:
  static const uint64_t MsgType = TMsgType;

  msg_protobuf_tx(uint64_t req_id)
    : req_id_(req_id)
  {
  }

  virtual uint64_t get_tx_msg_type() const
  {
    return TMsgType;
  }

  virtual uint64_t get_tx_req_id() const
  {
    return req_id_;
  }

  virtual uint64_t get_tx_header_len() const
  {
    return msg.ByteSize();
  }

  virtual void serialize_tx_header(void *dest)
  {
    msg.SerializeWithCachedSizesToArray(
        reinterpret_cast<google::protobuf::uint8 *>(dest));
  }

  virtual void message_sent()
  {
  }

  /* default to no body */
  virtual uint64_t get_tx_body_len() const
  {
    return 0;
  }

  virtual std::pair<const void *,size_t> next_tx_body_buf()
  {
    throw "Should not be called";
  }

  TMsg msg;

protected:
  uint64_t req_id_;
};

template <class TMsg, uint64_t TMsgType>
class msg_protobuf_rx : public message_rx {
public:
  static const uint64_t MsgType = TMsgType;

  msg_protobuf_rx(uint64_t req_id, size_t body_len)
    : req_id_(req_id), body_len_(body_len)
  {
  }

  virtual uint64_t get_msg_id() const
  {
    return req_id_;
  }

  virtual void header_received(const void *hdr, size_t hdr_len)
  {
    if (!msg.ParseFromArray(hdr, hdr_len))
      throw "parsing failed";
  }

  virtual std::pair<void *,size_t> next_body_rx_buf()
  {
    throw "Should not be called";
  }

  virtual void body_buf_received(size_t len)
  {
    throw "Should not be called";
  }

  virtual void rx_complete()
  {
  }

  TMsg msg;

protected:
  uint64_t req_id_;
  size_t body_len_;
};


class msg_load_model_req_tx :
  public msg_protobuf_tx<
    clockwork::ModelUploadRequest,
    clockwork::REQ_UPLOAD_MODEL>
{
public:
  msg_load_model_req_tx(uint64_t req_id)
    : msg_protobuf_tx(req_id)
  {
    body_send_state = BODY_SEND_SO;
  }

  /// setting blob lens in message
  void set_model_sizes()
  {
    msg.set_so_len(blob_so.size());
    msg.set_clockwork_len(blob_cw.size());
    msg.set_params_len(blob_cwparams.size());
  }

  virtual uint64_t get_tx_body_len() const
  {
    return blob_so.size() + blob_cw.size() + blob_cwparams.size();
  }

  virtual std::pair<const void *,size_t> next_tx_body_buf()
  {
    if (body_send_state == BODY_SEND_SO) {
      /* send shared object */
      body_send_state = BODY_SEND_CLOCKWORK;
      return std::make_pair(blob_so.data(), blob_so.size());
    } else if (body_send_state == BODY_SEND_CLOCKWORK) {
      /* send clockwork meta data */
      body_send_state = BODY_SEND_CWPARAMS;
      return std::make_pair(blob_cw.data(), blob_cw.size());
    } else if (body_send_state == BODY_SEND_CWPARAMS) {
      /* send clockwork paramaters */
      body_send_state = BODY_SEND_DONE;
      return std::make_pair(blob_cwparams.data(), blob_cwparams.size());
    } else {
      throw "TODO";
    }
  }

  std::string blob_so;
  std::string blob_cw;
  std::string blob_cwparams;
private:
  enum body_send_state {
    BODY_SEND_SO,
    BODY_SEND_CLOCKWORK,
    BODY_SEND_CWPARAMS,
    BODY_SEND_DONE,
  } body_send_state;
};

class msg_load_model_req_rx :
  public msg_protobuf_rx<
    clockwork::ModelUploadRequest,
    clockwork::REQ_UPLOAD_MODEL>
{
public:
  msg_load_model_req_rx(uint64_t req_id, size_t body_len)
    : msg_protobuf_rx(req_id, body_len), buf_so(0), buf_clockwork(0),
    buf_cwparams(0)
  {
    body_rx_state = BODY_RX_SO;
  }

  ~msg_load_model_req_rx()
  {
    if (buf_so)
      delete[] buf_so;
    if (buf_clockwork)
      delete[] buf_clockwork;
    if (buf_cwparams)
      delete[] buf_cwparams;
  }

  virtual void header_received(const void *hdr, size_t hdr_len)
  {
    msg_protobuf_rx::header_received(hdr, hdr_len);

    if (msg.so_len() + msg.clockwork_len() + msg.params_len() != body_len_)
      throw "model size sum does not match body len";

    buf_so = new uint8_t[msg.so_len()];
    buf_clockwork = new uint8_t[msg.clockwork_len()];
    buf_cwparams = new uint8_t[msg.params_len()];
  }

  virtual std::pair<void *,size_t> next_body_rx_buf()
  {
    if (body_rx_state == BODY_RX_SO) {
      body_rx_state = BODY_RX_CLOCKWORK;
      return std::make_pair(buf_so, msg.so_len());
    } else if (body_rx_state == BODY_RX_CLOCKWORK) {
      body_rx_state = BODY_RX_CWPARAMS;
      return std::make_pair(buf_clockwork, msg.clockwork_len());
    } else if (body_rx_state == BODY_RX_CWPARAMS) {
      body_rx_state = BODY_RX_DONE;
      return std::make_pair(buf_cwparams, msg.params_len());
    } else {
      throw "TODO";
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

  uint8_t *buf_so;
  uint8_t *buf_clockwork;
  uint8_t *buf_cwparams;

private:
  enum body_rx_state {
    BODY_RX_SO,
    BODY_RX_CLOCKWORK,
    BODY_RX_CWPARAMS,
    BODY_RX_DONE,
  } body_rx_state;

};


class msg_load_model_res_tx :
  public msg_protobuf_tx<
    clockwork::ModelUploadResponse,
    clockwork::RSP_UPLOAD_MODEL>
{
public:
  msg_load_model_res_tx(uint64_t req_id) :
    msg_protobuf_tx(req_id) { }
};

class msg_load_model_res_rx :
  public msg_protobuf_rx<
    clockwork::ModelUploadResponse,
    clockwork::RSP_UPLOAD_MODEL>
{
public:
  msg_load_model_res_rx(uint64_t req_id, size_t body_len)
    : msg_protobuf_rx(req_id, body_len) { }
};



class msg_inference_req_tx :
  public msg_protobuf_tx<
    clockwork::ModelInferenceRequest,
    clockwork::REQ_INFERENCE>
{
public:
  msg_inference_req_tx(uint64_t req_id)
    : msg_protobuf_tx(req_id), inputs_(0), inputs_size_(0) { }

  void set_inputs(void *inputs, size_t inputs_size)
  {
    inputs_ = inputs;
    inputs_size_ = inputs_size;
  }

  virtual uint64_t get_tx_body_len() const
  {
    return inputs_size_;
  }

  virtual std::pair<const void *,size_t> next_tx_body_buf()
  {
    return std::make_pair(inputs_, inputs_size_);
  }

private:
  void *inputs_;
  size_t inputs_size_;
};

class msg_inference_req_rx :
  public msg_protobuf_rx<
    clockwork::ModelInferenceRequest,
    clockwork::REQ_INFERENCE>
{
public:
  msg_inference_req_rx(uint64_t req_id, size_t body_len)
    : msg_protobuf_rx(req_id, body_len)
  {
    inputs_size_ = body_len;
    inputs_ = new uint8_t[body_len];
  }

  size_t get_inputs_size() const { return inputs_size_; }
  void *get_inputs() const { return inputs_; }

  virtual std::pair<void *,size_t> next_body_rx_buf()
  {
    return std::make_pair(inputs_, inputs_size_);
  }

  virtual void body_buf_received(size_t len)
  {
    if (len != inputs_size_)
      throw "unexpected body rx len";
  }

private:
  void *inputs_;
  size_t inputs_size_;
};


class msg_inference_res_tx :
  public msg_protobuf_tx<
    clockwork::ModelInferenceResponse,
    clockwork::RSP_INFERENCE>
{
public:
  msg_inference_res_tx(uint64_t req_id)
    : msg_protobuf_tx(req_id), outputs_(0), outputs_size_(0) { }

  void set_outputs(const void *outputs, size_t outputs_size)
  {
    outputs_ = outputs;
    outputs_size_ = outputs_size;
  }

  virtual uint64_t get_tx_body_len() const
  {
    return outputs_size_;
  }

  virtual std::pair<const void *,size_t> next_tx_body_buf()
  {
    return std::make_pair(outputs_, outputs_size_);
  }

private:
  const void *outputs_;
  size_t outputs_size_;
};

class msg_inference_res_rx :
  public msg_protobuf_rx<
    clockwork::ModelInferenceResponse,
    clockwork::RSP_INFERENCE>
{
public:
  msg_inference_res_rx(uint64_t req_id, size_t body_len)
    : msg_protobuf_rx(req_id, body_len)
  {
    outputs_size_ = body_len;
    outputs_ = new uint8_t[body_len];
  }

  void *get_outputs() const { return outputs_; }
  size_t get_outputs_size() const { return outputs_size_; }

  virtual std::pair<void *,size_t> next_body_rx_buf()
  {
    return std::make_pair(outputs_, outputs_size_);
  }

  virtual void body_buf_received(size_t len)
  {
    if (len != outputs_size_)
      throw "unexpected body rx len";
  }

private:
  void *outputs_;
  size_t outputs_size_;
};

#endif // ndef CLOCKWORK_NET_CLOCKWORK_H_
