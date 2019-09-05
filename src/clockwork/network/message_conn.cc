#include <iostream>
#include <boost/bind.hpp>
#include <clockwork/network.h>

message_connection::message_connection(asio::io_service& io_service,
    message_handler &handler)
  : socket_(io_service), resolver_(io_service),
    msg_rx_(message_receiver(this, handler))
{
}

/* establish outgoing connection */
void message_connection::connect(const std::string& server,
    const std::string& service)
{
  asio::ip::tcp::resolver::query query(server, service);
  resolver_.async_resolve(query,
      boost::bind(&message_connection::handle_resolved, this,
        asio::placeholders::error,
        asio::placeholders::iterator));
}

/* connection on socket established externally (e.g. through acceptor) */
void message_connection::established()
{
  /* disable nagle */
  asio::ip::tcp::no_delay option(true);
  socket_.set_option(option);

  msg_rx_.start();
  ready();
}

void message_connection::ready()
{
}

asio::ip::tcp::socket &message_connection::get_socket()
{
  return socket_;
}

void message_connection::handle_resolved(const asio::error_code& err,
    asio::ip::tcp::resolver::iterator endpoint_iterator)
{
  if (err) {
    abort_connection("Resolve failed");
    return;
  }

  asio::ip::tcp::endpoint endpoint = *endpoint_iterator;
  socket_.async_connect(endpoint,
      boost::bind(&message_connection::handle_established, this,
        asio::placeholders::error));
}

void message_connection::handle_established(const asio::error_code& err)
{
  if (err) {
    abort_connection("connect failed");
    return;
  }

  established();
}

void message_connection::abort_connection(const char *msg)
{
  /* todo */
  std::cerr << msg << std::endl;
  throw "todo";
}
