#ifndef _CLOCKWORK_NETWORK_WORKER_NET_H_
#define _CLOCKWORK_NETWORK_WORKER_NET_H_

#include <utility>
#include <cstring>
#include <string>
#include <iostream>
#include <boost/bind.hpp>
#include <asio.hpp>
#include "clockwork/network/network.h"
#include "clockwork/network/worker_api.h"

namespace clockwork {
namespace network {

using asio::ip::tcp;

class WorkerConnection : public message_connection, public message_handler {
private:
	ClockworkWorker* worker;
	message_sender msg_tx_;

public:
	WorkerConnection(asio::io_service &io_service, ClockworkWorker* worker) :
			message_connection(io_service, *this),
			msg_tx_(this, *this),
			worker(worker) {

	}

protected:
	virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
			uint64_t body_len, uint64_t msg_type, uint64_t msg_id) {
		switch (msg_type) {
			case load_model_from_disk_action_rx::MsgType: {}
		}
		return nullptr;
	}

  virtual void aborted_receive(message_connection *tcp_conn, message_rx *req) {
    delete req;
  }

  virtual void completed_receive(message_connection *tcp_conn, message_rx *req)
  {

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

};

class WorkerNetworkServer {
private:
	ClockworkWorker* worker;
	std::atomic_bool alive;
	std::thread network_thread;
	asio::io_service io_service;

public:
	WorkerNetworkServer(ClockworkWorker* worker, int port = 12345) :
			worker(worker), 
			network_thread(&WorkerNetworkServer::run, this, port),
			alive(true) {
	}

	void shutdown(bool awaitShutdown) {
		io_service.stop();
		if (awaitShutdown) {
			join();
		}
	}

	void join() {
		while (alive.load());
	}

	void run(int port) {
		try {
			tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port));
			start_accept(&acceptor);
			io_service.run();
		} catch (std::exception& e) {
			CHECK(false) << "Exception in network thread " << e.what();
		}
		alive.store(false);
	}

private:
	void start_accept(tcp::acceptor* acceptor) {
		message_connection *nc = new WorkerConnection(acceptor->get_io_service(), worker);

		acceptor->async_accept(nc->get_socket(),
			boost::bind(&WorkerNetworkServer::handle_accept, this, nc, acceptor,
				asio::placeholders::error));
	}

	void handle_accept(message_connection* nc, tcp::acceptor* acceptor, const asio::error_code& error) {
		if (!error) {
			nc->established();
		}

		start_accept(acceptor);
	}

};

}
}

#endif