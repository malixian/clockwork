#ifndef CLOCKWORK_NET_CLIENT_H_
#define CLOCKWORK_NET_CLIENT_H_

#include <clockwork/net_messages.h>

class client_rpc_base
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
class client_rpc : public client_rpc_base
{
public:
  client_rpc(TReq &req) : req_(req)
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
  virtual void request_done(client_rpc_base &req)
  {
  }

  void send_request(client_rpc_base &rb)
  {
    requests[rb.id()] = &rb;
    msg_tx_.send_request(rb.request());
  }

  virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
      uint64_t body_len, uint64_t msg_type, uint64_t msg_id)
  {
    client_rpc_base *rb = requests[msg_id];
    message_rx &mrx = rb->make_response(msg_type, body_len);

    return &mrx;
  }

  virtual void aborted_receive(message_connection *tcp_conn, message_rx *req)
  {
  }

  virtual void completed_receive(message_connection *tcp_conn, message_rx *req)
  {
    uint64_t id = req->get_msg_id();
    std::map<uint64_t, client_rpc_base *>::iterator it = requests.find(id);
    client_rpc_base *rb = it->second;
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
  std::map<uint64_t, client_rpc_base *> requests;
};


#endif // ndef CLOCKWORK_NET_CLIENT_H_
