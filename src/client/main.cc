#include <string>
#include <iostream>
#include <map>
#include <boost/bind.hpp>
#include <asio.hpp>

#include <clockwork/net_client.h>


struct model_description testmodel_a;


class benchmark_test_client :
  public clockwork_client_conn
{
public:
  benchmark_test_client(asio::io_service& io_service)
    : clockwork_client_conn(io_service)
  {
    inputs = new uint8_t[1024];
    msg_id = 1;
  }

  virtual void ready()
  {
    client_rpc<msg_load_model_req_tx, msg_load_model_res_rx> *lreq =
      new client_rpc<msg_load_model_req_tx, msg_load_model_res_rx>(msg_id++);

    lreq->req.set_model(testmodel_a.blob_a, testmodel_a.blob_a_len,
        testmodel_a.blob_b, testmodel_a.blob_b_len);
    lreq->req.msg.set_model_id(1);
    lreq->req.msg.set_batchsize(1);

    send_request(*lreq);


  }

  virtual void request_done(client_rpc_base &req)
  {
    delete &req;

    if (msg_id % 10000 == 0)
      std::cout << "Send request " << msg_id << std::endl;

    client_rpc<msg_inference_req_tx, msg_inference_res_rx> *ireq =
      new client_rpc<msg_inference_req_tx, msg_inference_res_rx>(msg_id++);
    ireq->req.msg.set_model_id(1);
    ireq->req.set_inputs(inputs, 1024);

    send_request(*ireq);

  }

private:
  void *inputs;
  uint64_t msg_id;
};

int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: client HOST PORT" << std::endl;
    return 1;
  }


  testmodel_a.blob_a_len = 32 * 1024;
  testmodel_a.blob_a = new uint8_t[testmodel_a.blob_a_len];
  testmodel_a.blob_b_len = 128 * 1024 * 1024;
  testmodel_a.blob_b = new uint8_t[testmodel_a.blob_b_len];
  size_t blob_b_len;
  void *blob_b;


  try
  {
    asio::io_service io_service;
    benchmark_test_client conn(io_service);
    conn.connect(argv[1], argv[2]);
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
