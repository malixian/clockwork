#include "clockwork/network/client.h"

namespace clockwork {
namespace network {
namespace client {

using asio::ip::tcp;
using namespace clockwork::clientapi;

Connection::Connection(asio::io_service& io_service): net_rpc_conn(io_service), ready_cb(nop_cb) {
}

void Connection::set_ready_cb(std::function<void()> cb) {
	ready_cb = cb;
}

void Connection::ready() {
	ready_cb();
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

void Connection::nop_cb() {}

}
}
}