#include <string>
#include <functional>
#include <iostream>
#include <map>
#include <boost/bind.hpp>
#include <asio.hpp>

#include <clockwork/net_client.h>

#include "clockwork/util.h"

using namespace clockwork::clientapi;


void client_test(ClientAPI &client)
{
  static std::string test_msg = std::string("Foo Bar!");

  InferenceRequest ireq;
  ireq.header.user_request_id = 42;
  ireq.model_id = 1337;
  ireq.batch_size = 1;
  ireq.input_size = test_msg.size();
  ireq.input = (void *) test_msg.c_str();

  client.infer(ireq, [](InferenceResponse &res) { std::cout << (char *) res.output << std::endl; });
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: client HOST PORT" << std::endl;
    return 1;
  }

  try
  {
    asio::io_service io_service;
    net_client conn(io_service);
    ClientAPI *api = &conn;
    conn.set_ready_cb([api](){ client_test(*api); });
    conn.connect(argv[1], argv[2]);
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }
  catch (const char * m)
  {
    std::cerr << m << std::endl;
  }

  return 0;
}
