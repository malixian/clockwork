#ifndef _CLOCKWORK_NETWORK_WORKER_H_
#define _CLOCKWORK_NETWORK_WORKER_H_

#include <utility>
#include <cstring>
#include <string>
#include <iostream>
#include <boost/bind.hpp>
#include <asio.hpp>
#include "clockwork/worker.h"
#include "clockwork/network/network.h"
#include "clockwork/network/worker_api.h"

namespace clockwork {
namespace network {
namespace worker {

using asio::ip::tcp;

/* Worker side of the Controller<>Worker API network impl.
A connection to the Clockwork Controller */
class Connection : public message_connection, public message_handler, public workerapi::Controller  {
private:
	ClockworkWorker* worker;
	message_sender msg_tx_;
	std::function<void(void)> on_close;

public:
	Connection(asio::io_service &io_service, ClockworkWorker* worker, std::function<void(void)> on_close);

protected:
	virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
			uint64_t body_len, uint64_t msg_type, uint64_t msg_id);

	virtual void aborted_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void closed();

public:
	
	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result);

};

/* Worker-side server that accepts connections from the Controller */
class Server : public workerapi::Controller {
private:
	ClockworkWorker* worker;
	std::atomic_bool alive;
	std::thread network_thread;
	asio::io_service io_service;

	Connection* current_connection;

public:
	Server(ClockworkWorker* worker, int port = 12345);

	void shutdown(bool awaitShutdown);
	void join();
	void run(int port);
	
	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result);

private:
	void start_accept(tcp::acceptor* acceptor);

	void handle_accept(Connection* connection, tcp::acceptor* acceptor, const asio::error_code& error);

};

}
}
}

#endif