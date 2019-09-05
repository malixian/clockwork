#ifndef CLOCKWORK_NETWORK_H_
#define CLOCKWORK_NETWORK_H_

#include <utility>
#include <boost/bind.hpp>
#include <asio.hpp>


class message_connection;

class message_tx {
public:
  virtual uint64_t get_tx_msg_type() const = 0;
  virtual uint64_t get_tx_req_id() const = 0;
  virtual uint64_t get_tx_header_len() const = 0;
  virtual uint64_t get_tx_body_len() const = 0;
  virtual void serialize_tx_header(void *dest) = 0;
  virtual void message_sent() = 0;
  virtual std::pair<const void *,size_t> next_tx_body_buf() = 0;
};


class message_rx {
public:
  virtual uint64_t get_msg_id() const = 0;
  virtual void header_received(const void *hdr, size_t hdr_len) = 0;
  virtual std::pair<void *,size_t> next_body_rx_buf() = 0;
  virtual void body_buf_received(size_t len) = 0;
  virtual void rx_complete() = 0;
};


class message_handler {
public:
  /* header length,  body length, message type, message id */
  virtual message_rx *new_rx_message(message_connection *tcp_conn,
      uint64_t header_len, uint64_t body_len, uint64_t msg_type,
      uint64_t msg_id) = 0;
  virtual void aborted_receive(message_connection *tcp_conn,
      message_rx *req) = 0;
  virtual void completed_receive(message_connection *tcp_conn,
      message_rx *req) = 0;

  virtual void aborted_transmit(message_connection *tcp_conn,
      message_tx *req) = 0;
  virtual void completed_transmit(message_connection *tcp_conn,
      message_tx *req) = 0;
};


class message_sender {
public:
  message_sender(message_connection *conn, message_handler &handler);
  void send_request(message_tx &req);
  void handle_prehdr_sent(const asio::error_code& error,
      size_t bytes_transferred);
  void handle_hdr_sent(const asio::error_code& error,
      size_t bytes_transferred);
  void handle_body_seg_sent(const asio::error_code& error,
      size_t bytes_transferred);
  void next_body_seg();
  void abort_connection(const char *msg);


  asio::ip::tcp::socket &socket_;

  static const size_t max_header_len = 1024;
  char header_buf[max_header_len];
  /* header length,  body length, message type, message id */
  uint64_t pre_header[4];

  message_connection *conn_;
  message_tx *req_;
  message_handler &handler_;
  size_t body_left;
  size_t body_seg_sent_;
};


class message_receiver {
public:
  message_receiver(message_connection *conn, message_handler &handler);
  void start();

private:
  void abort_connection(const char *msg);
  /* begin reading a new message */
  void read_new_message();
  /* common pre header received */
  void handle_pre_read(const asio::error_code& error,
      size_t bytes_transferred);
  /* header received */
  void handle_header_read(const asio::error_code& error,
      size_t bytes_transferred);
  /* body segment received */
  void handle_body_seg_read(const asio::error_code& error,
      size_t bytes_transferred);
  /* initiate rx for next body segment or finish message */
  void next_body_seg();


  asio::ip::tcp::socket &socket_;

  static const size_t max_header_len = 1024;
  char header_buf[max_header_len];
  /* header length,  body length, message type, message id */
  uint64_t pre_header[4];

  message_connection *conn_;
  message_handler &handler_;
  message_rx *req_;
  size_t body_left;
};


class message_connection {
public:
  message_connection(asio::io_service& io_service, message_handler &handler);
  /* establish outgoing connection */
  void connect(const std::string& server, const std::string& service);
  /* connection on socket established externally (e.g. through acceptor) */
  void established();
  asio::ip::tcp::socket &get_socket();

protected:
  virtual void ready();

private:
  void handle_resolved(const asio::error_code& err,
      asio::ip::tcp::resolver::iterator endpoint_iterator);
  void handle_established(const asio::error_code& err);
  void abort_connection(const char *msg);

  message_receiver msg_rx_;
  asio::ip::tcp::resolver resolver_;
  asio::ip::tcp::socket socket_;
};


#endif // ndef CLOCKWORK_NETWORK_H_
