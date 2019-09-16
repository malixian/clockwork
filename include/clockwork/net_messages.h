#ifndef CLOCKWORK_NET_CLOCKWORK_H_
#define CLOCKWORK_NET_CLOCKWORK_H_

#include <iostream>

#include <clockwork/network.h>
#include <clockwork.pb.h>


struct model_description {
  size_t blob_a_len;
  void *blob_a;
  size_t blob_b_len;
  void *blob_b;
};

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
    clockwork::ModelLoadRequest,
    clockwork::REQ_MODEL_LOAD>
{
public:
  msg_load_model_req_tx(uint64_t req_id)
    : msg_protobuf_tx(req_id), blob_a_(0), blob_a_len_(0), blob_b_(0),
    blob_b_len_(0)
  {
    body_send_state = BODY_SEND_BLOB_A;
  }

  void set_model(void *blob_a, size_t blob_a_len, void *blob_b,
      size_t blob_b_len)
  {
    blob_a_ = blob_a;
    blob_a_len_ = blob_a_len;
    msg.set_blob_a_len(blob_a_len);

    blob_b_ = blob_b;
    blob_b_len_ = blob_b_len;
    msg.set_blob_b_len(blob_b_len);
  }

  virtual uint64_t get_tx_body_len() const
  {
    return blob_a_len_ + blob_b_len_;
  }

  virtual std::pair<const void *,size_t> next_tx_body_buf()
  {
    if (body_send_state == BODY_SEND_BLOB_A) {
      body_send_state = BODY_SEND_BLOB_B;
      return std::make_pair(blob_a_, blob_a_len_);
    } else if (body_send_state == BODY_SEND_BLOB_B) {
      body_send_state = BODY_SEND_DONE;
      return std::make_pair(blob_b_, blob_b_len_);
    } else {
      throw "TODO";
    }
  }

private:
  enum body_send_state {
    BODY_SEND_BLOB_A,
    BODY_SEND_BLOB_B,
    BODY_SEND_DONE,
  } body_send_state;

  void *blob_a_;
  size_t blob_a_len_;
  void *blob_b_;
  size_t blob_b_len_;
};

class msg_load_model_req_rx :
  public msg_protobuf_rx<
    clockwork::ModelLoadRequest,
    clockwork::REQ_MODEL_LOAD>
{
public:
  msg_load_model_req_rx(uint64_t req_id, size_t body_len)
    : msg_protobuf_rx(req_id, body_len)
  {
    body_rx_state = BODY_RX_BLOB_A;
  }

  struct model_description &get_model() { return model_; }

  virtual void header_received(const void *hdr, size_t hdr_len)
  {
    msg_protobuf_rx::header_received(hdr, hdr_len);

    model_.blob_a_len = msg.blob_a_len();
    model_.blob_b_len = msg.blob_b_len();

    if (model_.blob_a_len + model_.blob_b_len != body_len_)
      throw "model size sum does not match body len";

    model_.blob_a = new uint8_t[model_.blob_a_len];
    model_.blob_b = new uint8_t[model_.blob_b_len];
  }

  virtual std::pair<void *,size_t> next_body_rx_buf()
  {
    if (body_rx_state == BODY_RX_BLOB_A) {
      body_rx_state = BODY_RX_BLOB_B;
      return std::make_pair(model_.blob_a, model_.blob_a_len);
    } else if (body_rx_state == BODY_RX_BLOB_B) {
      body_rx_state = BODY_RX_DONE;
      return std::make_pair(model_.blob_b, model_.blob_b_len);
    } else {
      throw "TODO";
    }
  }

  virtual void body_buf_received(size_t len)
  {
    size_t expected;

    if (body_rx_state == BODY_RX_BLOB_B) {
      expected = model_.blob_a_len;
    } else if (body_rx_state == BODY_RX_DONE) {
      expected = model_.blob_b_len;
    } else {
      throw "TODO";
    }

    if (expected != len)
      throw "unexpected body rx len";
  }

private:
  enum body_rx_state {
    BODY_RX_BLOB_A,
    BODY_RX_BLOB_B,
    BODY_RX_DONE,
  } body_rx_state;

  struct model_description model_;
};


class msg_load_model_res_tx :
  public msg_protobuf_tx<
    clockwork::ModelLoadResponse,
    clockwork::RES_MODEL_LOAD>
{
public:
  msg_load_model_res_tx(uint64_t req_id) :
    msg_protobuf_tx(req_id) { }
};

class msg_load_model_res_rx :
  public msg_protobuf_rx<
    clockwork::ModelLoadResponse,
    clockwork::RES_MODEL_LOAD>
{
public:
  msg_load_model_res_rx(uint64_t req_id, size_t body_len)
    : msg_protobuf_rx(req_id, body_len) { }
};



class msg_inference_req_tx :
  public msg_protobuf_tx<
    clockwork::ModelInferenceRequest,
    clockwork::REQ_MODEL_INFERENCE>
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
    clockwork::REQ_MODEL_INFERENCE>
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
    clockwork::RES_MODEL_INFERENCE>
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
    clockwork::RES_MODEL_INFERENCE>
{
public:
  msg_inference_res_rx(uint64_t req_id, size_t body_len)
    : msg_protobuf_rx(req_id, body_len)
  {
    outputs_size_ = body_len;
    outputs_ = new uint8_t[body_len];
  }

  void *get_outputs() const { return outputs_; }

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
