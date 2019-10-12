#ifndef _CLOCKWORK_NETWORK_WORKER_NET_H_
#define _CLOCKWORK_NETWORK_WORKER_NET_H_

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

using asio::ip::tcp;

/* Worker side of the Controller<>Worker API network impl */
class WorkerConnection : public message_connection, public message_handler, public workerapi::Controller  {
private:
	ClockworkWorker* worker;
	message_sender msg_tx_;

public:
	WorkerConnection(asio::io_service &io_service, ClockworkWorker* worker);

protected:
	virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
			uint64_t body_len, uint64_t msg_type, uint64_t msg_id);

	virtual void aborted_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req);

public:
	
	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result);

};


/* Controller side of the Controller<>Worker API network impl */
class ControllerConnection : public message_connection, public message_handler, public workerapi::Worker  {
private:
	workerapi::Controller* controller;
	message_sender msg_tx_;

public:
	std::atomic_bool connected;

	ControllerConnection(asio::io_service &io_service, workerapi::Controller* controller);

protected:
	virtual message_rx *new_rx_message(message_connection *tcp_conn, uint64_t header_len,
			uint64_t body_len, uint64_t msg_type, uint64_t msg_id);

	virtual void ready();

	virtual void aborted_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_receive(message_connection *tcp_conn, message_rx *req);

	virtual void completed_transmit(message_connection *tcp_conn, message_tx *req);

	virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req);

public:

	virtual void sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions);

	void sendAction(std::shared_ptr<workerapi::Action> &action);

};

class WorkerServer : public workerapi::Controller {
private:
	ClockworkWorker* worker;
	std::atomic_bool alive;
	std::thread network_thread;
	asio::io_service io_service;

	WorkerConnection* current_connection;

public:
	WorkerServer(ClockworkWorker* worker, int port = 12345);

	void shutdown(bool awaitShutdown);
	void join();
	void run(int port);
	
	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result);

private:
	void start_accept(tcp::acceptor* acceptor);

	void handle_accept(message_connection* nc, tcp::acceptor* acceptor, const asio::error_code& error);

};

class WorkerClient {
private:
	std::atomic_bool alive;
	asio::io_service io_service;
	std::thread network_thread;

public:
	WorkerClient();

	void run();

	void shutdown(bool awaitCompletion = false);

	void join();
	workerapi::Worker* connect(std::string host, std::string port, workerapi::Controller* controller);

};

}
}

#endif