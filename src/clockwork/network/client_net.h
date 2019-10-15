#ifndef _CLOCKWORK_NETWORK_CLIENT_NET_H_
#define _CLOCKWORK_NETWORK_CLIENT_NET_H_

#include <utility>
#include <cstring>
#include <string>
#include <iostream>
#include <boost/bind.hpp>
#include <asio.hpp>
#include "clockwork/worker.h"
#include "clockwork/network/network.h"
#include "clockwork/network/client_api.h"
#include "clockwork/network/net_rpc.h"

namespace clockwork {
namespace network {

using asio::ip::tcp;

/* Controller side of the Client<>Clockwork API network impl */
class ClockworkConnection : public message_connection, public message_handler  {
private:
	clientapi::ClientAPI* api;
	message_sender msg_tx_;

public:
	std::atomic_bool connected;

	ClockworkConnection(asio::io_service &io_service, clientapi::ClientAPI* api);

protected:
	virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
			uint64_t body_len, uint64_t msg_type, uint64_t msg_id);

	virtual void ready();
	virtual void closed();

	virtual void aborted_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req);

};

class ControllerServer {
private:
	clientapi::ClientAPI* api;
	std::atomic_bool alive;
	std::thread network_thread;
	asio::io_service io_service;

public:
	ControllerServer(clientapi::ClientAPI* api, int port = 12346);

	void shutdown(bool awaitShutdown);
	void join();
	void run(int port);

private:
	void start_accept(tcp::acceptor* acceptor);

	void handle_accept(ClockworkConnection* connection, tcp::acceptor* acceptor, const asio::error_code& error);

};

class ControllerClient: public net_rpc_conn, public clientapi::ClientAPI {
public:
  ControllerClient(asio::io_service& io_service): net_rpc_conn(io_service), ready_cb(nop_cb) {}

  void set_ready_cb(std::function<void()> cb)
  {
    ready_cb = cb;
  }

  virtual void ready()
  {
    ready_cb();
  }

  virtual void request_done(net_rpc_base &req)
  {
    delete &req;

  }

  virtual void uploadModel(clientapi::UploadModelRequest &request,
      std::function<void(clientapi::UploadModelResponse&)> callback)
  {
    auto rpc = new net_rpc<msg_upload_model_req_tx, msg_upload_model_rsp_rx>(
      [callback](msg_upload_model_rsp_rx &rsp) {
        clientapi::UploadModelResponse response;
        rsp.get(response);
        callback(response);
      }
    );

    rpc->req.set(request);
    send_request(*rpc);
  }

  virtual void infer(clientapi::InferenceRequest &request,
      std::function<void(clientapi::InferenceResponse&)> callback)
  {
  	auto rpc = new net_rpc_receive_payload<msg_inference_req_tx, msg_inference_rsp_rx>(
  		[callback](msg_inference_rsp_rx &rsp) {
        clientapi::InferenceResponse response;
        rsp.get(response);
        callback(response);
      }
  	);

    rpc->req.set(request);
    send_request(*rpc);
  }

  virtual void evict(clientapi::EvictRequest &request,
      std::function<void(clientapi::EvictResponse&)> callback)
  {
    auto rpc = new net_rpc<msg_evict_req_tx, msg_evict_rsp_rx>(
      [callback](msg_evict_rsp_rx &rsp) {
        clientapi::EvictResponse response;
        rsp.get(response);
        callback(response);
      }
    );

    rpc->req.set(request);
    send_request(*rpc);
  }

  /** This is a 'backdoor' API function for ease of experimentation */
  virtual void loadRemoteModel(clientapi::LoadModelFromRemoteDiskRequest &request,
      std::function<void(clientapi::LoadModelFromRemoteDiskResponse&)> callback)
  {
    auto rpc = new net_rpc<msg_load_remote_model_req_tx, msg_load_remote_model_rsp_rx>(
      [callback](msg_load_remote_model_rsp_rx &rsp) {
        clientapi::LoadModelFromRemoteDiskResponse response;
        rsp.get(response);
        callback(response);
      }
    );

    rpc->req.set(request);
    send_request(*rpc);
  }

private:
  std::function<void()> ready_cb;

  static void nop_cb() { }
};

	
}
}

#endif