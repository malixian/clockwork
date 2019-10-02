#ifndef CLOCKWORK_NET_CLIENT_H_
#define CLOCKWORK_NET_CLIENT_H_

#include <functional>

#include <clockwork/net_rpc.h>
#include <clockwork/api/client_api.h>

class net_client  :
  public net_rpc_conn, public clockwork::clientapi::ClientAPI
{
public:
  net_client(asio::io_service& io_service)
    : net_rpc_conn(io_service), ready_cb(nop_cb)
  {
    msg_id = 1;
  }

  void set_ready_cb(std::function<void()> cb)
  {
    ready_cb = cb;
  }

  virtual void ready()
  {
    ready_cb();
  }

  virtual void request_done(net_rpc_base &req)
  {
    delete &req;

  }

  virtual void uploadModel(clockwork::clientapi::UploadModelRequest &request,
      std::function<void(clockwork::clientapi::UploadModelResponse&)> callback)
  {
    throw "todo";
  }

  virtual void infer(clockwork::clientapi::InferenceRequest &request,
      std::function<void(clockwork::clientapi::InferenceResponse&)> callback)
  {
    uint64_t usr_id = request.header.user_request_id;
    auto ireq = new net_rpc<msg_inference_req_tx, msg_inference_res_rx>(msg_id++,
        [usr_id, callback](msg_inference_res_rx &res) {
          clockwork::clientapi::InferenceResponse ir;
          ir.header.user_request_id = usr_id;
          ir.model_id = res.msg.model_id();
          ir.batch_size = res.msg.batch_size();
          ir.output = res.get_outputs();
          ir.output_size = res.get_outputs_size();
          callback(ir);
        });

    ireq->req.msg.set_model_id(request.model_id);
    ireq->req.msg.set_batch_size(request.batch_size);
    ireq->req.set_inputs(request.input, request.input_size);

    send_request(*ireq);
  }

  /** This is a 'backdoor' API function for ease of experimentation */
  virtual void loadRemoteModel(clockwork::clientapi::LoadModelFromRemoteDiskRequest &request,
      std::function<void(clockwork::clientapi::LoadModelFromRemoteDiskResponse&)> callback)
  {
    throw "todo";
  }

private:
  std::function<void()> ready_cb;
  uint64_t msg_id;

  static void nop_cb() { }
};

#endif // ndef CLOCKWORK_NET_CLIENT_H_
