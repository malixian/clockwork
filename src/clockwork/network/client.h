#ifndef _CLOCKWORK_NETWORK_CLIENT_H_
#define _CLOCKWORK_NETWORK_CLIENT_H_

#include <atomic>
#include <string>
#include <asio.hpp>
#include "clockwork/worker.h"
#include "clockwork/network/network.h"
#include "clockwork/network/rpc.h"
#include "clockwork/network/client_api.h"

namespace clockwork {
namespace network {
namespace client {

using asio::ip::tcp;
using namespace clockwork::clientapi;

/* Client side of the Client<>Controller API network impl.
Represents a connection of a client to the Clockwork controller */
class Connection: public net_rpc_conn, public ClientAPI {
public:
  Connection(asio::io_service& io_service);

  void set_ready_cb(std::function<void()> cb);

  // net_rpc_conn methods
  virtual void ready();
  virtual void request_done(net_rpc_base &req);

  // clientapi methods
  virtual void uploadModel(UploadModelRequest &request, std::function<void(UploadModelResponse&)> callback);
  virtual void infer(InferenceRequest &request, std::function<void(InferenceResponse&)> callback);
  virtual void evict(EvictRequest &request, std::function<void(EvictResponse&)> callback);
  virtual void loadRemoteModel(LoadModelFromRemoteDiskRequest &request, std::function<void(LoadModelFromRemoteDiskResponse&)> callback);

private:
  std::function<void()> ready_cb;

  static void nop_cb();
};

}
}
}

#endif