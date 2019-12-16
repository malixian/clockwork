#include "clockwork/network/client.h"

namespace clockwork {
namespace network {
namespace client {

using asio::ip::tcp;
using namespace clockwork::clientapi;

Connection::Connection(asio::io_service& io_service): net_rpc_conn(io_service), connected(false) {
}

void Connection::ready() {
	connected.store(true);
}

void Connection::request_done(net_rpc_base &req) {
	delete &req;
}

void Connection::uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback) {
	auto rpc = new net_rpc<msg_upload_model_req_tx, msg_upload_model_rsp_rx>(
	  [callback](msg_upload_model_rsp_rx &rsp) {
	    UploadModelResponse response;
	    rsp.get(response);
	    callback(response);
	  }
	);

	rpc->req.set(request);
	send_request(*rpc);
}

void Connection::infer(InferenceRequest &request, std::function<void(InferenceResponse&)> callback) {
	auto rpc = new net_rpc_receive_payload<msg_inference_req_tx, msg_inference_rsp_rx>(
		[callback](msg_inference_rsp_rx &rsp) {
			InferenceResponse response;
			rsp.get(response);
			callback(response);
		});

	rpc->req.set(request);
	send_request(*rpc);
}

void Connection::evict(EvictRequest &request, std::function<void(EvictResponse&)> callback){
	auto rpc = new net_rpc<msg_evict_req_tx, msg_evict_rsp_rx>(
		[callback](msg_evict_rsp_rx &rsp) {
			EvictResponse response;
			rsp.get(response);
			callback(response);
		});

	rpc->req.set(request);
	send_request(*rpc);
}

  /** This is a 'backdoor' API function for ease of experimentation */
void Connection::loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback) {
	auto rpc = new net_rpc<msg_load_remote_model_req_tx, msg_load_remote_model_rsp_rx>(
		[callback](msg_load_remote_model_rsp_rx &rsp) {
			LoadModelFromRemoteDiskResponse response;
			rsp.get(response);
			callback(response);
		});

	rpc->req.set(request);
	send_request(*rpc);
}

ConnectionManager::ConnectionManager() : alive(true), network_thread(&ConnectionManager::run, this) {}

void ConnectionManager::run() {
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

void ConnectionManager::shutdown(bool awaitCompletion) {
	alive.store(false);
	io_service.stop();
	if (awaitCompletion) {
		join();
	}
}

void ConnectionManager::join() {
	network_thread.join();
}

Connection* ConnectionManager::connect(std::string host, std::string port) {
	try {
		Connection* c = new Connection(io_service);
		c->connect(host, port);
		std::cout << "Connecting to clockwork @ " << host << ":" << port << std::endl;
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
	return nullptr;
}

}
}
}