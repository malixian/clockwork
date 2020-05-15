#ifndef _CLOCKWORK_NETWORK_CONTROLLER_H_
#define _CLOCKWORK_NETWORK_CONTROLLER_H_

#include <atomic>
#include <string>
#include <asio.hpp>
#include "clockwork/worker.h"
#include "clockwork/network/network.h"
#include "clockwork/network/worker_api.h"
#include "clockwork/network/client_api.h"

namespace clockwork {
namespace network {
namespace controller {

using asio::ip::tcp;

class WorkerConnection : public workerapi::Worker {
public:

	virtual void sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions) = 0;
	virtual void sendAction(std::shared_ptr<workerapi::Action> action) = 0;
};

/* Controller side of the Controller<>Worker API network impl.
Represents a connection to a single worker. */
class SingleWorkerConnection : public message_connection, public message_handler  {
private:
	message_sender msg_tx_;
	workerapi::Controller* controller;

public:
	std::atomic_bool connected;

	SingleWorkerConnection(asio::io_service &io_service, 
		workerapi::Controller* controller,
		tbb::concurrent_queue<message_tx*> &queue);

	virtual void send_message();

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

class BondedWorkerConnection : public WorkerConnection {
public:
	tbb::concurrent_queue<message_tx*> queue;
	std::vector<SingleWorkerConnection*> connections;

	BondedWorkerConnection() {}

	virtual void sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions);

	void sendAction(std::shared_ptr<workerapi::Action> action);

private:
	void send(message_tx* tx);

};

/* WorkerManager is used to connect to multiple workers.
Connect can be called multiple times, to connect to multiple workers.
Each WorkerConnection will handle a single worker.
The WorkerManager internally has just one IO thread to handle IO for all connections */
class WorkerManager {
private:
	std::atomic_bool alive;
	asio::io_service io_service;
	std::thread network_thread;

public:
	WorkerManager();

	void run();

	void shutdown(bool awaitCompletion = false);

	void join();
	WorkerConnection* connect(std::string host, std::string port, workerapi::Controller* controller);
	WorkerConnection* connect(std::string host, std::vector<std::string> ports, workerapi::Controller* controller);

};

/* Controller side of the Client<>Controller API network impl */
class ClientConnection : public message_connection, public message_handler  {
private:
	clientapi::ClientAPI* api;
	tbb::concurrent_queue<message_tx*> queue;
	message_sender msg_tx_;

public:
	std::atomic_bool connected;

	ClientConnection(asio::io_service &io_service, clientapi::ClientAPI* api);

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


/* Controller-side server for the Client API.  
Accepts connections and requests from users/clients.
Creates ClientConnection for incoming client connections.
The Server internally maintains one IO thread for all client connections. */
class Server {
private:
	clientapi::ClientAPI* api;
	std::atomic_bool alive;
	asio::io_service io_service;
	std::thread network_thread;

public:
	Server(clientapi::ClientAPI* api, int port = 12346);

	void shutdown(bool awaitShutdown);
	void join();
	void run(int port);

private:
	void start_accept(tcp::acceptor* acceptor);

	void handle_accept(ClientConnection* connection, tcp::acceptor* acceptor, const asio::error_code& error);

};

}
}
}

#endif