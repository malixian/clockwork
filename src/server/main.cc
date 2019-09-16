#include <utility>
#include <string>
#include <iostream>
#include <boost/bind.hpp>
#include <asio.hpp>

#include <clockwork/net_messages.h>

using asio::ip::tcp;


class clockwork_server_conn :
  public message_connection, public message_handler
{
public:
  clockwork_server_conn(asio::io_service& io_service)
    : message_connection(io_service, *this), msg_tx_(this, *this)
  {
  }

protected:
  virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
      uint64_t body_len, uint64_t msg_type, uint64_t msg_id)
  {
    switch (msg_type) {
      case msg_load_model_req_rx::MsgType:
        return new msg_load_model_req_rx(msg_id, body_len);
      case msg_inference_req_rx::MsgType:
        return new msg_inference_req_rx(msg_id, body_len);
      default:
        throw "todo";
    }
  }

  virtual void aborted_receive(message_connection *tcp_conn, message_rx *req)
  {
    delete req;
  }

  virtual void completed_receive(message_connection *tcp_conn, message_rx *req)
  {
    msg_load_model_req_rx *load_req;
    msg_inference_req_rx *inf_req;

    if (load_req = dynamic_cast<msg_load_model_req_rx *>(req)) {
      // TODO: actually do load request

      msg_load_model_res_tx *res =
        new msg_load_model_res_tx(load_req->get_msg_id());
      res->msg.set_status(0);

      msg_tx_.send_message(*res);
    } else if (inf_req = dynamic_cast<msg_inference_req_rx *>(req)) {
      // TODO: actually do inference

      msg_inference_res_tx *res =
        new msg_inference_res_tx(inf_req->get_msg_id());
      res->msg.set_status(0);

      msg_tx_.send_message(*res);
    } else {
      throw "todo";
    }

    delete req;
  }

  virtual void completed_transmit(message_connection *tcp_conn, message_tx *req)
  {
    delete req;
  }

  virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req)
  {
    std::cout << "aborted response" << std::endl;
    delete req;
  }

  message_sender msg_tx_;
};



class tcp_server
{
public:
  tcp_server(asio::io_service& io_service, uint16_t port)
    : acceptor_(io_service, tcp::endpoint(tcp::v4(), port))
  {
    start_accept();
  }

protected:
  message_connection *new_connection()
  {
    return new clockwork_server_conn(acceptor_.get_io_service());
  }

private:
  void start_accept()
  {
    message_connection *nc = new_connection();

    acceptor_.async_accept(nc->get_socket(),
        boost::bind(&tcp_server::handle_accept, this, nc,
          asio::placeholders::error));
  }

  void handle_accept(message_connection *nc,
      const asio::error_code& error)
  {
    if (!error) {
      nc->established();
    }

    start_accept();
  }

  tcp::acceptor acceptor_;
};

int main()
{
  try
  {
    asio::io_service io_service;
    tcp_server server(io_service, 12345);
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
