#include "clockwork/network/worker.h"
#include "clockwork/util.h"
#include <sstream>
#include "clockwork/thread.h"

namespace clockwork {
namespace network {
namespace worker {

using asio::ip::tcp;

bool verbose = false;


class infer_action_rx_using_io_pool : public infer_action_rx {
private:
  clockwork::MemoryPool* host_io_pool;


public:
  infer_action_rx_using_io_pool(clockwork::MemoryPool* host_io_pool):
    infer_action_rx(), host_io_pool(host_io_pool) {
  }

  virtual ~infer_action_rx_using_io_pool() {
    delete static_cast<uint8_t*>(body_);
  }

  virtual void get(workerapi::Infer &action) {
  	infer_action_rx::get(action);

  	// Copy the input body into a cached page
    action.input = host_io_pool->alloc(body_len_);
    CHECK(action.input != nullptr) << "Unable to alloc from host_io_pool for infer action input";
    std::memcpy(action.input, body_, body_len_);
  }
};

class infer_result_tx_using_io_pool : public infer_result_tx {
private:
  clockwork::MemoryPool* host_io_pool;

public:
  infer_result_tx_using_io_pool(clockwork::MemoryPool* host_io_pool):
    infer_result_tx(), host_io_pool(host_io_pool) {
  }

  virtual ~infer_result_tx_using_io_pool() {
  	delete static_cast<uint8_t*>(body_);
  }

  virtual void set(workerapi::InferResult &result) {
  	// Memory allocated with cudaMallocHost doesn't play nicely with asio.
  	// Until we solve it, just do a memcpy here :(
  	infer_result_tx::set(result);
    body_ = new uint8_t[result.output_size];
    std::memcpy(body_, result.output, result.output_size);
    host_io_pool->free(result.output);
  }

};

class InferUsingIOPool : public workerapi::Infer {
public:
	clockwork::MemoryPool* host_io_pool;
	InferUsingIOPool(clockwork::MemoryPool* host_io_pool):
		host_io_pool(host_io_pool) {}
	~InferUsingIOPool() {
		host_io_pool->free(input);
	}
};

Connection::Connection(asio::io_service &io_service, ClockworkWorker* worker, 
			std::function<void(void)> on_close,
			tbb::concurrent_queue<message_tx*> &queue,
			ConnectionStats &stats) :
		message_connection(io_service, *this),
		msg_tx_(this, *this, queue),
		worker(worker),
		stats(stats),
		on_close(on_close),
		alive(true) {
}

class StatTracker {
public:
	uint64_t previous_value = 0;
	uint64_t update(std::atomic_uint64_t &counter) {
		uint64_t current_value = counter.load();
		uint64_t delta = current_value - previous_value;
		previous_value = current_value;
		return delta;
	}
};

message_rx* Connection::new_rx_message(message_connection *tcp_conn, uint64_t header_len,
		uint64_t body_len, uint64_t msg_type, uint64_t msg_id) {
	using namespace clockwork::workerapi;

	if (msg_type == ACT_LOAD_MODEL_FROM_DISK) {
		auto msg = new load_model_from_disk_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_LOAD_WEIGHTS) {
		auto msg = new load_weights_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_INFER) {
		auto msg = new infer_action_rx_using_io_pool(worker->runtime->manager->host_io_pool);
		msg->set_body_len(body_len);
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_GET_WORKER_STATE) {
		auto msg = new get_worker_state_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_EVICT_WEIGHTS) {
		auto msg = new evict_weights_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	} else if (msg_type == ACT_CLEAR_CACHE) {
		auto msg = new clear_cache_action_rx();
		msg->set_msg_id(msg_id);
		return msg;
	}
	
	CHECK(false) << "Unsupported msg_type " << msg_type;
	return nullptr;
}

void Connection::aborted_receive(message_connection *tcp_conn, message_rx *req) {
	delete req;
}

void Connection::completed_receive(message_connection *tcp_conn, message_rx *req) {
	std::vector<std::shared_ptr<workerapi::Action>> actions;

	if (auto load_model = dynamic_cast<load_model_from_disk_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::LoadModelFromDisk>();
		load_model->get(*action);
		actions.push_back(action);

		if (!verbose) std::cout << "Received " << actions[0]->str() << std::endl;

	} else if (auto load_weights = dynamic_cast<load_weights_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::LoadWeights>();
		load_weights->get(*action);
		actions.push_back(action);

		stats.load++;
	} else if (auto infer = dynamic_cast<infer_action_rx_using_io_pool*>(req)) {
		auto action = std::make_shared<InferUsingIOPool>(worker->runtime->manager->host_io_pool);
		infer->get(*action);
		actions.push_back(action);

		stats.infer++;
	} else if (auto evict = dynamic_cast<evict_weights_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::EvictWeights>();
		evict->get(*action);
		actions.push_back(action);

		stats.evict++;
	} else if (auto clear_cache = dynamic_cast<clear_cache_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::ClearCache>();
		clear_cache->get(*action);
		actions.push_back(action);
	} else if (auto get_worker_state = dynamic_cast<get_worker_state_action_rx*>(req)) {
		auto action = std::make_shared<workerapi::GetWorkerState>();
		get_worker_state->get(*action);
		actions.push_back(action);

		if (!verbose) std::cout << "Received " << actions[0]->str() << std::endl;

	} else {
		CHECK(false) << "Received an unsupported message_rx type";
	}
	if (verbose) std::cout << "Received " << actions[0]->str() << std::endl;

	actions[0]->clock_delta = estimate_clock_delta();	

	stats.total_pending++;

	delete req;
	worker->sendActions(actions);
}

void Connection::completed_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void Connection::aborted_transmit(message_connection *tcp_conn, message_tx *req) {
	delete req;
}

void Connection::send_message() {
	msg_tx_.send_message();
}

void Connection::closed() {
	alive = false;
	this->on_close();
}

Server::Server(ClockworkWorker* worker, int port) :
		is_started(false),
		worker(worker),
		io_service(),
		stats(),
		network_thread(&Server::run, this, port),
		printer(&Server::print, this) {
	threading::initNetworkThread(network_thread);
	threading::initLoggerThread(printer);
}

Server::~Server() {}

void Server::shutdown(bool awaitShutdown) {
	io_service.stop();
	if (awaitShutdown) {
		join();
	}
}

void Server::join() {
	while (!is_started);
	network_thread.join();
}

void Server::run(int port) {
	try {
		is_started.store(true);
		std::vector<tcp::acceptor*> acceptors;
		for (unsigned i = 0; i < 8; i++) {
			auto endpoint = tcp::endpoint(tcp::v4(), port+i);
			auto acceptor = new tcp::acceptor(io_service, endpoint);
			acceptors.push_back(acceptor);
			start_accept(acceptor);
			std::cout << "IO service thread listening on " << endpoint << std::endl;
		}
		io_service.run();
	} catch (std::exception& e) {
		CHECK(false) << "Exception in network thread: " << e.what();
	} catch (const char* m) {
		CHECK(false) << "Exception in network thread: " << m;
	}
	std::cout << "Server exiting" << std::endl;
}

void Server::send(message_tx* msg) {
	queue.push(msg);
	for (auto &p : connections) {
		p.second->send_message();
	}
}

// workerapi::Controller::sendResult
void Server::sendResult(std::shared_ptr<workerapi::Result> result) {
	if (verbose) std::cout << "Sending " << result->str() << std::endl;
	using namespace workerapi;
	if (auto load_model = std::dynamic_pointer_cast<LoadModelFromDiskResult>(result)) {
		auto tx = new load_model_from_disk_result_tx();
		tx->set(*load_model);
		send(tx);

		if (!verbose) std::cout << "Sending " << result->str() << std::endl;
	} else if (auto load_weights = std::dynamic_pointer_cast<LoadWeightsResult>(result)) {
		auto tx = new load_weights_result_tx();
		tx->set(*load_weights);
		send(tx);

	} else if (auto infer = std::dynamic_pointer_cast<InferResult>(result)) {
		auto tx = new infer_result_tx_using_io_pool(worker->runtime->manager->host_io_pool);
		tx->set(*infer);
		send(tx);

	} else if (auto evict_weights = std::dynamic_pointer_cast<EvictWeightsResult>(result)) {
		auto tx = new evict_weights_result_tx();
		tx->set(*evict_weights);
		send(tx);

	} else if (auto clear_cache = std::dynamic_pointer_cast<ClearCacheResult>(result)) {
		auto tx = new clear_cache_result_tx();
		tx->set(*clear_cache);
		send(tx);

	} else if (auto get_worker_state = std::dynamic_pointer_cast<GetWorkerStateResult>(result)) {
		auto tx = new get_worker_state_result_tx();
		tx->set(*get_worker_state);
		send(tx);

		if (!verbose) std::cout << "Sending " << result->str() << std::endl;
	} else if (auto error = std::dynamic_pointer_cast<ErrorResult>(result)) {
		auto tx = new error_result_tx();
		tx->set(*error);
		send(tx);

		stats.errors++;
	} else {
		CHECK(false) << "Sending an unsupported result type";
	}

	stats.total_pending--;
}	

void Server::print() {
	// uint64_t print_every = 10000000000UL; // 10s
	// uint64_t last_print = util::now();

	// StatTracker load;
	// StatTracker evict;
	// StatTracker infer;
	// StatTracker errors;

	// while (alive) {
	// 	uint64_t now = util::now();
	// 	if (last_print + print_every > now) {
	// 		usleep(200000); // 200ms sleep
	// 		continue;
	// 	}

	// 	uint64_t dload = load.update(stats.load);
	// 	uint64_t dinfer = infer.update(stats.infer);
	// 	uint64_t devict = evict.update(stats.evict);

	// 	uint64_t pending = stats.total_pending;
	// 	uint64_t derrors = errors.update(stats.errors);

	// 	std::stringstream s;
	// 	s << "Clock Skew=" << estimate_clock_delta()
	// 	  << "  RTT=" << estimate_rtt()
	// 	  << "  LdWts=" << dload
	// 	  << "  Inf=" << dinfer
	// 	  << "  Evct=" << devict
	// 	  << "  || Total Pending=" << pending
	// 	  << "  Errors=" << derrors
	// 	  << std::endl;

	// 	std::cout << s.str();
	// 	last_print = now;
	// }
}

void Server::start_accept(tcp::acceptor* acceptor) {
	int connection_id = connection_id_seed++;
	auto connection = new Connection(acceptor->get_io_service(), worker, [this, connection_id]{
		auto it = connections.find(connection_id);
		if (it != connections.end()) {
			auto connection = it->second;
			connections.erase(it);
			delete connection;
		}
	}, queue, stats);	

	acceptor->async_accept(connection->get_socket(),
		boost::bind(&Server::handle_accept, this, connection_id, connection, acceptor,
			asio::placeholders::error));
}

void Server::handle_accept(int connection_id, Connection* connection, tcp::acceptor* acceptor, const asio::error_code& error) {
	if (error) {
		throw std::runtime_error(error.message());
	}

	connection->established();

	connections[connection_id] = connection;

	start_accept(acceptor);
}

}
}
}
