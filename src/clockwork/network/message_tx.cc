#include <iostream>
#include <boost/bind.hpp>
#include <clockwork/network.h>

message_sender::message_sender(message_connection *conn, message_handler &handler)
  : socket_(conn->get_socket()), conn_(conn), handler_(handler), req_(0)
{
}

void message_sender::send_request(message_tx &req)
{
  if (!req_) {
    start_send(req);
  } else {
    tx_queue_.push_back(&req);
  }
}

void message_sender::send_next_message()
{
  if (tx_queue_.empty())
    return;
  message_tx *req = tx_queue_.front();
  tx_queue_.pop_front();
  start_send(*req);
}

void message_sender::start_send(message_tx &req)
{
  /* header length,  body length, message type, message id */
  pre_header[0] = req.get_tx_header_len();
  pre_header[1] = req.get_tx_body_len();
  pre_header[2] = req.get_tx_msg_type();
  pre_header[3] = req.get_tx_req_id();

  req.serialize_tx_header(header_buf);

  req_ = &req;
  asio::async_write(socket_, asio::buffer(pre_header),
      boost::bind(&message_sender::handle_prehdr_sent, this,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred));
}

void message_sender::handle_prehdr_sent(const asio::error_code& error,
    size_t bytes_transferred)
{
  if (bytes_transferred != 32) {
    abort_connection("Invalid number of bytes sent for header lengths");
    return;
  }

  asio::async_write(socket_,
      asio::buffer(header_buf, req_->get_tx_header_len()),
      boost::bind(&message_sender::handle_hdr_sent, this,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred));
}

void message_sender::handle_hdr_sent(const asio::error_code& error,
    size_t bytes_transferred)
{
  if (bytes_transferred != req_->get_tx_header_len()) {
    abort_connection("Invalid number of bytes sent for header");
    return;
  }

  body_left = req_->get_tx_body_len();
  next_body_seg();
}

void message_sender::handle_body_seg_sent(const asio::error_code& error,
    size_t bytes_transferred)
{
  if (bytes_transferred != body_seg_sent_) {
    abort_connection("Invalid number of bytes sent for body segment");
    return;
  }

  body_left -= bytes_transferred;
  next_body_seg();
}

/* initiate rx for next body segment or finish request */
void message_sender::next_body_seg()
{
  /* if we have a body... */
  if (body_left > 0) {
    std::pair<const void *,size_t> body_buf = req_->next_tx_body_buf();
    assert(body_left >= body_buf.second);

    body_seg_sent_ = body_buf.second;
    asio::async_write(socket_, asio::buffer(body_buf.first, body_buf.second),
        boost::bind(&message_sender::handle_body_seg_sent, this,
          asio::placeholders::error,
          asio::placeholders::bytes_transferred));
  } else {
    req_->message_sent();
    handler_.completed_transmit(conn_, req_);
    req_ = 0;

    send_next_message();
  }
}

void message_sender::abort_connection(const char *msg)
{
  /* todo */
  std::cerr << msg << std::endl;
  throw "todo";
}
