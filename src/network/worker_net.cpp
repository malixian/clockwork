#include <utility>
#include <cstring>
#include <string>
#include <iostream>
#include <boost/bind.hpp>
#include <asio.hpp>

#include <clockwork/net_messages.h>
//#include <clockwork/model.h>
//#include "clockwork/util.h"
#include <clockwork/api/client_api.h>

using asio::ip::tcp;
using namespace clockwork::clientapi;

class dummy_client : public ClientAPI {
public:
  virtual void uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback)
  {
    std::cerr << "uploadModel()" << std::endl;
  }

  virtual void infer(InferenceRequest &request, std::function<void(InferenceResponse&)> callback)
  {
    std::cerr << "infer()" << std::endl;
    InferenceResponse ir;
    ir.header.status = 0;
    ir.header.user_request_id = request.header.user_request_id;
    ir.model_id = request.model_id;
    ir.batch_size = request.batch_size;

    static const std::string msg = std::string("Hello World!");
    ir.output_size = msg.size() + 1;
    ir.output = new char[ir.output_size];
    std::strcpy((char *) ir.output, msg.c_str());
    callback(ir);
  }

  virtual void evict(EvictRequest &request, std::function<void(EvictResponse&)> callback)
  {
    std::cerr << "evict()" << std::endl;
  }

  virtual void loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback)
  {
    std::cerr << "loadRemoteModel()" << std::endl;
  }

};


class clockwork_server_conn :
  public message_connection, public message_handler
{
public:
  clockwork_server_conn(asio::io_service& io_service, ClientAPI &client)
    : message_connection(io_service, *this), msg_tx_(this, *this),
      client_(client)
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

    if (inf_req = dynamic_cast<msg_inference_req_rx *>(req)) {
      InferenceRequest *api_req = new InferenceRequest;
      api_req->model_id = inf_req->msg.model_id();
      api_req->batch_size = inf_req->msg.batch_size();
      api_req->input_size = inf_req->get_inputs_size();
      api_req->input = inf_req->get_inputs();

      uint64_t msg_id = inf_req->get_msg_id();
      client_.infer(*api_req, [api_req, msg_id, this](InferenceResponse& res) {
            delete api_req;

            msg_inference_res_tx *ir_tx = new msg_inference_res_tx(msg_id);
            ir_tx->msg.mutable_header()->set_status(res.header.status);
            ir_tx->msg.set_model_id(res.model_id);
            ir_tx->msg.set_batch_size(res.batch_size);
            ir_tx->set_outputs(res.output, res.output_size);
            msg_tx_.send_message(*ir_tx);
          });
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

  ClientAPI &client_;
  message_sender msg_tx_;
};



class tcp_server
{
public:
  tcp_server(asio::io_service& io_service, uint16_t port, ClientAPI &client)
    : acceptor_(io_service, tcp::endpoint(tcp::v4(), port)), client_(client)
  {
    start_accept();
  }

protected:
  message_connection *new_connection()
  {
    return new clockwork_server_conn(acceptor_.get_io_service(), client_);
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
  ClientAPI &client_;
};

int main()
{
  try
  {
    dummy_client dc;

    asio::io_service io_service;
    tcp_server server(io_service, 12345, dc);
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
