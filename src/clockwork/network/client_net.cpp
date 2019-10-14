#include "clockwork/network/client_net.h"

namespace clockwork{
namespace network{

ClockworkConnection::ClockworkConnection(asio::io_service &io_service, clientapi::ClientAPI* api) :
		message_connection(io_service, *this),
		msg_tx_(this, *this),
		api(api),
		connected(false) {
}

message_rx* ClockworkConnection::new_rx_message(message_connection *tcp_conn, uint64_t header_len,
		uint64_t body_len, uint64_t msg_type, uint64_t msg_id) {
	using namespace clockwork::clientapi;

	if (msg_type == REQ_INFERENCE) {
		auto msg = new msg_inference_req_rx();
		msg->set_body_len(body_len);
		msg->set_msg_id(msg_id);
		return msg;

	} else if (msg_type == REQ_EVICT) {
		auto msg = new msg_evict_req_rx();
		msg->set_msg_id(msg_id);
		return msg;

	} else if (msg_type == REQ_LOAD_REMOTE_MODEL) {
		auto msg = new msg_load_remote_model_req_rx();
		msg->set_msg_id(msg_id);
		return msg;

	} else if (msg_type == REQ_UPLOAD_MODEL) {
		auto msg = new msg_upload_model_req_rx();
		msg->set_body_len(body_len);
		msg->set_msg_id(msg_id);
		return msg;

	}
	
	CHECK(false) << "Unsupported msg_type " << msg_type;
	return nullptr;
}

void ClockworkConnection::ready() {
	connected.store(true);
}

void ClockworkConnection::closed() {
}

void ClockworkConnection::aborted_receive(message_connection *tcp_conn, message_rx *req) {
	delete req;
}

void ClockworkConnection::completed_receive(message_connection *tcp_conn, message_rx *req) {

	if (auto infer = dynamic_cast<msg_inference_req_rx*>(req)) {
		auto request = new clientapi::InferenceRequest();
		infer->get(*request);
		api->infer(*request, [this, request] (clientapi::InferenceResponse &response) {
			delete request;
			auto rsp = new msg_inference_rsp_tx();
			rsp->set(response);
			msg_tx_.send_message(*rsp);
		});

	} else if (auto evict = dynamic_cast<msg_evict_req_rx*>(req)) {
		auto request = new clientapi::EvictRequest();
		evict->get(*request);
		api->evict(*request, [this, request] (clientapi::EvictResponse &response) {
			delete request;
			auto rsp = new msg_evict_rsp_tx();
			rsp->set(response);
			msg_tx_.send_message(*rsp);
		});

	} else if (auto load_model = dynamic_cast<msg_load_remote_model_req_rx*>(req)) {
		auto request = new clientapi::LoadModelFromRemoteDiskRequest();
		load_model->get(*request);
		api->loadRemoteModel(*request, [this, request] (clientapi::LoadModelFromRemoteDiskResponse &response) {
			delete request;
			auto rsp = new msg_load_remote_model_rsp_tx();
			rsp->set(response);
			msg_tx_.send_message(*rsp);
		});
		
	} else if (auto upload_model = dynamic_cast<msg_upload_model_req_rx*>(req)) {
		auto request = new clientapi::UploadModelRequest();
		upload_model->get(*request);
		api->uploadModel(*request, [this, request] (clientapi::UploadModelResponse &response) {
			delete request;
			auto rsp = new msg_upload_model_rsp_tx();
			rsp->set(response);
			msg_tx_.send_message(*rsp);
		});
		
	} else {
		CHECK(false) << "Received an unsupported RPC request";
	}

	delete req;
}

void ClockworkConnection::completed_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void ClockworkConnection::aborted_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

ControllerServer::ControllerServer(clientapi::ClientAPI* api, int port) :
		api(api),
		network_thread(&ControllerServer::run, this, port),
		alive(true) {
}

void ControllerServer::shutdown(bool awaitShutdown) {
	io_service.stop();
	if (awaitShutdown) {
		join();
	}
}

void ControllerServer::join() {
	while (alive.load());
}

void ControllerServer::run(int port) {
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
	std::cout << "ControllerServer exiting" << std::endl;
	alive.store(false);
}

void ControllerServer::start_accept(tcp::acceptor* acceptor) {
	auto connection = new ClockworkConnection(acceptor->get_io_service(), api);

	acceptor->async_accept(connection->get_socket(),
		boost::bind(&ControllerServer::handle_accept, this, connection, acceptor,
			asio::placeholders::error));
}

void ControllerServer::handle_accept(ClockworkConnection* connection, tcp::acceptor* acceptor, const asio::error_code& error) {
	if (error) {
		throw std::runtime_error(error.message());
	}

	connection->established();
	start_accept(acceptor);
}

	
}
}