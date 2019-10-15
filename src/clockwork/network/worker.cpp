#include "clockwork/network/worker.h"

namespace clockwork {
namespace network {
namespace worker {

using asio::ip::tcp;

Connection::Connection(asio::io_service &io_service, ClockworkWorker* worker, std::function<void(void)> on_close) :
		message_connection(io_service, *this),
		msg_tx_(this, *this),
		worker(worker),
		on_close(on_close) {

}

message_rx* Connection::new_rx_message(message_connection *tcp_conn, uint64_t header_len,
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

void Connection::aborted_receive(message_connection *tcp_conn, message_rx *req) {
	delete req;
}

void Connection::completed_receive(message_connection *tcp_conn, message_rx *req) {
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

	std::cout << "Received " << actions[0]->str() << std::endl;

	delete req;
	worker->sendActions(actions);
}

void Connection::completed_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void Connection::aborted_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void Connection::sendResult(std::shared_ptr<workerapi::Result> result) {
	std::cout << "Sending " << result->str() << std::endl;

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

void Connection::closed() {
	this->on_close();
}

Server::Server(ClockworkWorker* worker, int port) :
		worker(worker), 
		network_thread(&Server::run, this, port),
		alive(true) {
}

void Server::shutdown(bool awaitShutdown) {
	io_service.stop();
	if (awaitShutdown) {
		join();
	}
}

void Server::join() {
	while (alive.load());
}

void Server::run(int port) {
	try {
		tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port));
		start_accept(&acceptor);
		std::cout << "Running io service thread" << std::endl;
		io_service.run();
	} catch (std::exception& e) {
		CHECK(false) << "Exception in network thread: " << e.what();
	} catch (const char* m) {
		CHECK(false) << "Exception in network thread: " << m;
	}
	std::cout << "Server exiting" << std::endl;
	alive.store(false);
}

// workerapi::Controller::sendResult
void Server::sendResult(std::shared_ptr<workerapi::Result> result) {
	if (current_connection == nullptr) {
		std::cout << "Dropping result " << result->str() << std::endl;
	} else {
		current_connection->sendResult(result);
	}
}	

void Server::start_accept(tcp::acceptor* acceptor) {
	auto connection = new Connection(acceptor->get_io_service(), worker, [this]{
		this->current_connection = nullptr;
		delete this->current_connection;
	});

	acceptor->async_accept(connection->get_socket(),
		boost::bind(&Server::handle_accept, this, connection, acceptor,
			asio::placeholders::error));
}

void Server::handle_accept(Connection* connection, tcp::acceptor* acceptor, const asio::error_code& error) {
	if (error) {
		throw std::runtime_error(error.message());
	}

	connection->established();
	this->current_connection = connection;
	start_accept(acceptor);
}

}
}
}