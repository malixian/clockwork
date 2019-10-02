#include <iostream>
#include <boost/bind.hpp>
#include <clockwork/network.h>


message_receiver::message_receiver(message_connection *conn, message_handler &handler)
  : socket_(conn->get_socket()), handler_(handler), conn_(conn)
{
}

void message_receiver::start()
{
  read_new_message();
}

void message_receiver::abort_connection(const char *msg)
{
  std::cerr << "aborting connection: " << msg << std::endl;
  // TODO
}

/* begin reading a new message */
void message_receiver::read_new_message()
{
  /* begin by reading the pre-header */
  asio::async_read(socket_, asio::buffer(pre_header),
      boost::bind(&message_receiver::handle_pre_read, this,
      asio::placeholders::error,
      asio::placeholders::bytes_transferred));
}

/* common pre header received */
void message_receiver::handle_pre_read(const asio::error_code& error,
    size_t bytes_transferred)
{
  if (error) {
    std::cerr << error << std::endl;
    abort_connection("Error on pre header read");
    return;
  }

  if (bytes_transferred != 32) {
    abort_connection("Invalid number of bytes read for header lengths");
    return;
  }

  if (pre_header[0] > max_header_len) {
    abort_connection("Specified header length larger than supported maximum");
    return;
  }

  asio::async_read(socket_, asio::buffer(header_buf, pre_header[0]),
      boost::bind(&message_receiver::handle_header_read, this,
      asio::placeholders::error,
      asio::placeholders::bytes_transferred));
}

/* header received */
void message_receiver::handle_header_read(const asio::error_code& error,
    size_t bytes_transferred)
{
  if (error) {
    abort_connection("Error on pre header read");
    return;
  }

  if (bytes_transferred != pre_header[0]) {
    abort_connection("Header incomplete");
    return;
  }

  req_ = handler_.new_rx_message(conn_, pre_header[0], pre_header[1],
      pre_header[2], pre_header[3]);
  req_->header_received(header_buf, pre_header[0]);

  body_left = pre_header[1];
  next_body_seg();
}

/* body segment received */
void message_receiver::handle_body_seg_read(const asio::error_code& error,
    size_t bytes_transferred)
{
  if (error) {
    abort_connection("Error on pre header read");
    return;
  }

  req_->body_buf_received(bytes_transferred);

  body_left -= bytes_transferred;
  next_body_seg();
}

/* initiate rx for next body segment or finish message */
void message_receiver::next_body_seg()
{
  /* if we have a body... */
  if (body_left > 0) {
    std::pair<void *,size_t> body_buf = req_->next_body_rx_buf();
    size_t len = std::min(body_buf.second, body_left);

    asio::async_read(socket_, asio::buffer(body_buf.first, len),
        boost::bind(&message_receiver::handle_body_seg_read, this,
          asio::placeholders::error,
          asio::placeholders::bytes_transferred));
  } else {
    req_->rx_complete();
    handler_.completed_receive(conn_, req_);
    req_ = 0;
    read_new_message();
  }
}
