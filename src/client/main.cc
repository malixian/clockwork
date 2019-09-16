#include <string>
#include <iostream>
#include <map>
#include <boost/bind.hpp>
#include <asio.hpp>

#include <clockwork/net_messages.h>


struct model_description testmodel_a;

class RequestBase
{
public:
  uint64_t id()
  {
    return request().get_tx_req_id();
  }

  virtual message_tx &request() = 0;
  virtual message_rx &make_response(uint64_t msg_type, uint64_t body_len) = 0;
  virtual void done() = 0;
};

template<class TReq, class TRes>
class Request : public RequestBase
{
public:
  Request(TReq &req) : req_(req)
  {
  }

  virtual message_tx &request()
  {
    return req_;
  }

  virtual message_rx &make_response(uint64_t msg_type, uint64_t body_len)
  {
    if (msg_type != TRes::MsgType)
      throw "unexpected message type in response";

    res_ = new TRes(id(), body_len);
    return *res_;
  }

  virtual void done()
  {
    std::cerr << "request done:" << id() << std::endl;
  }

  TReq &req_;
  TRes *res_;
};


class clockwork_client_conn :
  public message_connection, public message_handler
{
public:
  clockwork_client_conn(asio::io_service& io_service)
    : message_connection(io_service, *this), msg_tx_(this, *this)
  {
  }

protected:
  virtual void ready()
  {
    msg_load_model_req_tx *ltxmsg = new msg_load_model_req_tx(42, testmodel_a, 1, 1);
    Request<msg_load_model_req_tx, msg_load_model_res_rx> *lreq =
      new Request<msg_load_model_req_tx, msg_load_model_res_rx>(*ltxmsg);
    send_request(*lreq);

    void *inputs = new uint8_t[1024];
    msg_inference_req_tx *txmsg = new msg_inference_req_tx(43, 1, inputs, 1024);
    Request<msg_inference_req_tx, msg_inference_res_rx> *req =
      new Request<msg_inference_req_tx, msg_inference_res_rx>(*txmsg);
    send_request(*req);
  }

  void send_request(RequestBase &rb)
  {
    requests[rb.id()] = &rb;
    msg_tx_.send_request(rb.request());
  }

  virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
      uint64_t body_len, uint64_t msg_type, uint64_t msg_id)
  {
    RequestBase *rb = requests[msg_id];
    message_rx &mrx = rb->make_response(msg_type, body_len);

    return &mrx;
  }

  virtual void aborted_receive(message_connection *tcp_conn, message_rx *req)
  {
  }

  virtual void completed_receive(message_connection *tcp_conn, message_rx *req)
  {
    uint64_t id = req->get_msg_id();
    std::map<uint64_t, RequestBase *>::iterator it = requests.find(id);
    RequestBase *rb = it->second;
    requests.erase(it);
    rb->done();
  }

  virtual void completed_transmit(message_connection *tcp_conn, message_tx *req)
  {
  }

  virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req)
  {
  }

  message_sender msg_tx_;
  std::map<uint64_t, RequestBase *> requests;
};


int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: client HOST PORT" << std::endl;
    return 1;
  }


  testmodel_a.blob_a_len = 32 * 1024;
  testmodel_a.blob_a = new uint8_t[testmodel_a.blob_a_len];
  testmodel_a.blob_b_len = 128 * 1024 * 1024;
  testmodel_a.blob_b = new uint8_t[testmodel_a.blob_b_len];
  size_t blob_b_len;
  void *blob_b;


  try
  {
    asio::io_service io_service;
    clockwork_client_conn conn(io_service);
    conn.connect(argv[1], argv[2]);
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
