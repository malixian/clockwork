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

class WorkerConnection : public message_connection, public message_handler, public workerapi::Controller  {
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
		using namespace clockwork::workerapi;

		if (msg_type == ACT_LOAD_MODEL_FROM_DISK) {
			auto msg = new load_model_from_disk_action_rx();
			msg->set_msg_id(msg_id);
			return msg;
		} else if (msg_type == ACT_LOAD_WEIGHTS) {
			auto msg = new load_weights_action_rx();
			msg->set_msg_id(msg_id);
			return msg;
		} else if (msg_type == ACT_INFER) {
			auto msg = new infer_action_rx();
			msg->set_body_len(body_len);
			msg->set_msg_id(msg_id);
			return msg;
		} else if (msg_type == ACT_EVICT_WEIGHTS) {
			auto msg = new evict_weights_action_rx();
			msg->set_msg_id(msg_id);
			return msg;
		}
		
		CHECK(false) << "Unsupported msg_type " << msg_type;
		return nullptr;
	}

	virtual void aborted_receive(message_connection *tcp_conn, message_rx *req) {
		delete req;
	}

	virtual void completed_receive(message_connection *tcp_conn, message_rx *req) {
		std::vector<std::shared_ptr<workerapi::Action>> actions;

		if (auto load_model = dynamic_cast<load_model_from_disk_action_rx*>(req)) {
			auto action = std::make_shared<workerapi::LoadModelFromDisk>();
			load_model->get(*action);
			actions.push_back(action);
		} else if (auto load_weights = dynamic_cast<load_weights_action_rx*>(req)) {
			auto action = std::make_shared<workerapi::LoadWeights>();
			load_weights->get(*action);
			actions.push_back(action);
		} else if (auto infer = dynamic_cast<infer_action_rx*>(req)) {
			auto action = std::make_shared<workerapi::Infer>();
			infer->get(*action);
			actions.push_back(action);
		} else if (auto evict = dynamic_cast<evict_weights_action_rx*>(req)) {
			auto action = std::make_shared<workerapi::EvictWeights>();
			evict->get(*action);
			actions.push_back(action);
		} else {
			CHECK(false) << "Received an unsupported message_rx type";
		}

		delete req;
		worker->sendActions(actions);

	}

	virtual void completed_transmit(message_connection *tcp_conn, message_tx *req) {
		delete req;
	}

	virtual void aborted_transmit(message_connection *tcp_conn, message_tx *req) {
		delete req;
	}

public:
	
	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result) {
		using namespace workerapi;
		if (auto load_model = std::dynamic_pointer_cast<LoadModelFromDiskResult>(result)) {
			auto tx = new load_model_from_disk_result_tx();
			tx->set(*load_model);
			msg_tx_.send_message(*tx);
		} else if (auto load_weights = std::dynamic_pointer_cast<LoadWeightsResult>(result)) {
			auto tx = new load_weights_result_tx();
			tx->set(*load_weights);
			msg_tx_.send_message(*tx);
		} else if (auto infer = std::dynamic_pointer_cast<InferResult>(result)) {
			auto tx = new infer_result_tx();
			tx->set(*infer);
			msg_tx_.send_message(*tx);
		} else if (auto evict_weights = std::dynamic_pointer_cast<EvictWeightsResult>(result)) {
			auto tx = new evict_weights_result_tx();
			tx->set(*evict_weights);
			msg_tx_.send_message(*tx);
		} else if (auto error = std::dynamic_pointer_cast<ErrorResult>(result)) {
			auto tx = new error_result_tx();
			tx->set(*error);
			msg_tx_.send_message(*tx);
		} else {
			CHECK(false) << "Sending an unsupported result type";
		}
	}	

};

class WorkerServer : public workerapi::Controller {
private:
	ClockworkWorker* worker;
	std::atomic_bool alive;
	std::thread network_thread;
	asio::io_service io_service;

	WorkerConnection* current_connection;

public:
	WorkerServer(ClockworkWorker* worker, int port = 12345) :
			worker(worker), 
			network_thread(&WorkerServer::run, this, port),
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
	
	// workerapi::Controller::sendResult
	virtual void sendResult(std::shared_ptr<workerapi::Result> result) {
		current_connection->sendResult(result);
	}	

private:
	void start_accept(tcp::acceptor* acceptor) {
		current_connection = new WorkerConnection(acceptor->get_io_service(), worker);

		acceptor->async_accept(current_connection->get_socket(),
			boost::bind(&WorkerServer::handle_accept, this, current_connection, acceptor,
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