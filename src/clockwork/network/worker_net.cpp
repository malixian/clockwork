#include "clockwork/network/worker_net.h"

namespace clockwork {
namespace network {

using asio::ip::tcp;

WorkerConnection::WorkerConnection(asio::io_service &io_service, ClockworkWorker* worker) :
		message_connection(io_service, *this),
		msg_tx_(this, *this),
		worker(worker) {

}

message_rx* WorkerConnection::new_rx_message(message_connection *tcp_conn, uint64_t header_len,
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

void WorkerConnection::aborted_receive(message_connection *tcp_conn, message_rx *req) {
	delete req;
}

void WorkerConnection::completed_receive(message_connection *tcp_conn, message_rx *req) {
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

void WorkerConnection::completed_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void WorkerConnection::aborted_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void WorkerConnection::sendResult(std::shared_ptr<workerapi::Result> result) {
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


ControllerConnection::ControllerConnection(asio::io_service &io_service, workerapi::Controller* controller) :
		message_connection(io_service, *this),
		msg_tx_(this, *this),
		controller(controller),
		connected(false) {
}

message_rx* ControllerConnection::new_rx_message(message_connection *tcp_conn, uint64_t header_len,
		uint64_t body_len, uint64_t msg_type, uint64_t msg_id) {
	using namespace clockwork::workerapi;

	if (msg_type == RES_ERROR) {
		auto msg = new error_result_rx();
		msg->set_msg_id(msg_id);
		return msg;

	} else if (msg_type == RES_LOAD_MODEL_FROM_DISK) {
		auto msg = new load_model_from_disk_result_rx();
		msg->set_msg_id(msg_id);
		return msg;

	} else if (msg_type == RES_LOAD_WEIGHTS) {
		auto msg = new load_weights_result_rx();
		msg->set_msg_id(msg_id);
		return msg;

	} else if (msg_type == RES_INFER) {
		auto msg = new infer_result_rx();
		msg->set_body_len(body_len);
		msg->set_msg_id(msg_id);
		return msg;

	} else if (msg_type == RES_EVICT_WEIGHTS) {
		auto msg = new evict_weights_result_rx();
		msg->set_msg_id(msg_id);
		return msg;

	}
	
	CHECK(false) << "Unsupported msg_type " << msg_type;
	return nullptr;
}

void ControllerConnection::ready() {
	connected.store(true);
}

void ControllerConnection::aborted_receive(message_connection *tcp_conn, message_rx *req) {
	delete req;
}

void ControllerConnection::completed_receive(message_connection *tcp_conn, message_rx *req) {
	if (auto error = dynamic_cast<error_result_rx*>(req)) {
		auto result = std::make_shared<workerapi::ErrorResult>();
		error->get(*result);
		controller->sendResult(result);

	} else if (auto load_model = dynamic_cast<load_model_from_disk_result_rx*>(req)) {
		auto result = std::make_shared<workerapi::LoadModelFromDiskResult>();
		load_model->get(*result);
		controller->sendResult(result);

	} else if (auto load_weights = dynamic_cast<load_weights_result_rx*>(req)) {
		auto result = std::make_shared<workerapi::LoadWeightsResult>();
		load_weights->get(*result);
		controller->sendResult(result);
		
	} else if (auto infer = dynamic_cast<infer_result_rx*>(req)) {
		auto result = std::make_shared<workerapi::InferResult>();
		infer->get(*result);
		controller->sendResult(result);
		
	} else if (auto evict = dynamic_cast<evict_weights_result_rx*>(req)) {
		auto result = std::make_shared<workerapi::EvictWeightsResult>();
		evict->get(*result);
		controller->sendResult(result);
		
	} else {
		CHECK(false) << "Received an unsupported message_rx type";
	}

	delete req;
}

void ControllerConnection::completed_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void ControllerConnection::aborted_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void ControllerConnection::sendActions(std::vector<std::shared_ptr<workerapi::Action>> &actions) {
	for (auto &action : actions) {
		sendAction(action);
	}
}

void ControllerConnection::sendAction(std::shared_ptr<workerapi::Action> &action) {
	std::cout << "Sending " << action->str() << std::endl;

	if (auto load_model = std::dynamic_pointer_cast<workerapi::LoadModelFromDisk>(action)) {
		auto tx = new load_model_from_disk_action_tx();
		tx->set(*load_model);
		msg_tx_.send_message(*tx);

	} else if (auto load_weights = std::dynamic_pointer_cast<workerapi::LoadWeights>(action)) {
		auto tx = new load_weights_action_tx();
		tx->set(*load_weights);
		msg_tx_.send_message(*tx);

	} else if (auto infer = std::dynamic_pointer_cast<workerapi::Infer>(action)) {
		auto tx = new infer_action_tx();
		tx->set(*infer);
		msg_tx_.send_message(*tx);

	} else if (auto evict_weights = std::dynamic_pointer_cast<workerapi::EvictWeights>(action)) {
		auto tx = new evict_weights_action_tx();
		tx->set(*evict_weights);
		msg_tx_.send_message(*tx);

	} else {
		CHECK(false) << "Sending an unsupported action type";
	}
}

WorkerServer::WorkerServer(ClockworkWorker* worker, int port) :
		worker(worker), 
		network_thread(&WorkerServer::run, this, port),
		alive(true) {
}

void WorkerServer::shutdown(bool awaitShutdown) {
	io_service.stop();
	if (awaitShutdown) {
		join();
	}
}

void WorkerServer::join() {
	while (alive.load());
}

void WorkerServer::run(int port) {
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
	alive.store(false);
}

// workerapi::Controller::sendResult
void WorkerServer::sendResult(std::shared_ptr<workerapi::Result> result) {
	current_connection->sendResult(result);
}	

void WorkerServer::start_accept(tcp::acceptor* acceptor) {
	auto connection = new WorkerConnection(acceptor->get_io_service(), worker);

	acceptor->async_accept(connection->get_socket(),
		boost::bind(&WorkerServer::handle_accept, this, connection, acceptor,
			asio::placeholders::error));
}

void WorkerServer::handle_accept(WorkerConnection* connection, tcp::acceptor* acceptor, const asio::error_code& error) {
	if (error) {
		throw std::runtime_error(error.message());
	}

	connection->established();
	this->current_connection = connection;
	start_accept(acceptor);
}

WorkerClient::WorkerClient() : alive(true), network_thread(&WorkerClient::run, this) {}

void WorkerClient::run() {
	while (alive) {
		try {
			asio::io_service::work work(io_service);
			io_service.run();
		} catch (std::exception& e) {
			alive.store(false);
			CHECK(false) << "Exception in network thread: " << e.what();
		} catch (const char* m) {
			alive.store(false);
			CHECK(false) << "Exception in network thread: " << m;
		}
	}
}

void WorkerClient::shutdown(bool awaitCompletion) {
	alive.store(false);
	io_service.stop();
	if (awaitCompletion) {
		join();
	}
}

void WorkerClient::join() {
	network_thread.join();
}

workerapi::Worker* WorkerClient::connect(std::string host, std::string port, workerapi::Controller* controller) {
	try {
		ControllerConnection* c = new ControllerConnection(io_service, controller);
		c->connect(host, port);
		std::cout << "Connecting to worker " << host << ":" << port << std::endl;
		while (alive.load() && !c->connected.load()); // If connection fails, alive sets to false
		std::cout << "Connection established" << std::endl;
		return c;
	} catch (std::exception& e) {
		alive.store(false);
		io_service.stop();
		CHECK(false) << "Exception in network thread: " << e.what();
	} catch (const char* m) {
		alive.store(false);
		io_service.stop();
		CHECK(false) << "Exception in network thread: " << m;
	}
}

}
}