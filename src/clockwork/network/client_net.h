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
	
}
}

#endif