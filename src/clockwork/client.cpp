#include <cstring>
#include <future>
#include <atomic>
#include <dmlc/logging.h>
#include "clockwork/client.h"
#include "clockwork/api/client_api.h"
#include "clockwork/network/client.h"

namespace clockwork
{

class NetworkClient : public Client
{
public:
	bool print;
	std::atomic_int request_id_seed;
	int user_id;
	network::client::ConnectionManager *manager;
	network::client::Connection *connection;
	ModelSet models;

	NetworkClient(network::client::ConnectionManager *manager, network::client::Connection *connection, bool print);
	virtual ~NetworkClient();

	virtual Model *get_model(int model_id);
	virtual std::future<Model *> get_model_async(int model_id);
	virtual Model *upload_model(std::vector<uint8_t> &serialized_model);
	virtual std::future<Model *> upload_model_async(std::vector<uint8_t> &serialized_model);
	virtual Model *load_remote_model(std::string model_path);
	virtual std::future<Model *> load_remote_model_async(std::string model_path);
	virtual ModelSet ls();
	virtual std::future<ModelSet> ls_async();
};

class ModelImpl : public Model
{
public:
	bool print;
	NetworkClient *client;

	const int model_id_;
	const std::string source_;
	const int input_size_;

	ModelImpl(NetworkClient *client, int model_id, std::string source, int input_size, bool print);

	virtual int id();
	virtual int input_size();
	virtual std::string source();

	virtual std::vector<uint8_t> infer(std::vector<uint8_t> &input);
	virtual std::future<std::vector<uint8_t>> infer_async(std::vector<uint8_t> &input);
	virtual void evict();
	virtual std::future<void> evict_async();
};

NetworkClient::NetworkClient(network::client::ConnectionManager *manager, network::client::Connection *connection, bool print) : manager(manager), connection(connection), user_id(0), request_id_seed(0), print(print)
{
}

NetworkClient::~NetworkClient()
{
	// TODO: delete connection and manager
}

Model *NetworkClient::get_model(int model_id)
{
	return get_model_async(model_id).get();
}

std::future<Model *> NetworkClient::get_model_async(int model_id)
{
	// TODO: implement this properly or remove
	auto promise = std::make_shared<std::promise<Model *>>();

	if (models.find(model_id) == models.end()) {
		try {
			this->models = ls();
		} catch (const std::runtime_error &e) {
			promise->set_exception(std::make_exception_ptr(e));
		}
	}

	if (models.find(model_id) != models.end()) {
		promise->set_value(models[model_id]);
	} else {
		promise->set_exception(std::make_exception_ptr(clockwork_error("Model does not exist")));
	}

	return promise->get_future();
}

Model *NetworkClient::upload_model(std::vector<uint8_t> &serialized_model)
{
	return upload_model_async(serialized_model).get();
}

std::future<Model *> NetworkClient::upload_model_async(std::vector<uint8_t> &serialized_model)
{
	auto promise = std::make_shared<std::promise<Model *>>();
	promise->set_exception(std::make_exception_ptr(clockwork_error("upload_model not implemented")));
	return promise->get_future();
}

Model *NetworkClient::load_remote_model(std::string model_path)
{
	return load_remote_model_async(model_path).get();
}

std::future<Model *> NetworkClient::load_remote_model_async(std::string model_path)
{
	auto promise = std::make_shared<std::promise<Model *>>();

	clientapi::LoadModelFromRemoteDiskRequest load_model;
	load_model.header.user_id = 0;
	load_model.header.user_request_id = request_id_seed++;
	load_model.remote_path = model_path;

	if (print) std::cout << "<--  " << load_model.str() << std::endl;

	connection->loadRemoteModel(load_model, [this, promise, model_path](clientapi::LoadModelFromRemoteDiskResponse &response) {
		if (print) std::cout << " --> " << response.str() << std::endl;
		if (response.header.status == clockworkSuccess)
		{
			promise->set_value(new ModelImpl(this, response.model_id, model_path, response.input_size, print));
		}
		else if (response.header.status == clockworkInitializing)
		{
			promise->set_exception(std::make_exception_ptr(clockwork_initializing(response.header.message)));
		} 
		else
		{
			promise->set_exception(std::make_exception_ptr(clockwork_error(response.header.message)));
		}
	});

	return promise->get_future();
}

ModelSet NetworkClient::ls() {
	return ls_async().get();
}

std::future<ModelSet> NetworkClient::ls_async()
{
	auto promise = std::make_shared<std::promise<ModelSet>>();

	clientapi::LSRequest ls;
	ls.header.user_id = 0;
	ls.header.user_request_id = request_id_seed++;

	if (print) std::cout << "<--  " << ls.str() << std::endl;

	connection->ls(ls, [this, promise](clientapi::LSResponse &response) {
		if (print) std::cout << " --> " << response.str() << std::endl;
		if (response.header.status == clockworkSuccess)
		{
			ModelSet models;
			for (auto &model : response.models) {
				auto impl = new ModelImpl(this, model.model_id, model.remote_path, model.input_size, print);
				models[model.model_id] = impl;
			}
			promise->set_value(models);
		}
		else if (response.header.status == clockworkInitializing)
		{
			promise->set_exception(std::make_exception_ptr(clockwork_initializing(response.header.message)));
		} 
		else
		{
			promise->set_exception(std::make_exception_ptr(clockwork_error(response.header.message)));
		}
	});

	return promise->get_future();
}

ModelImpl::ModelImpl(NetworkClient *client, int model_id, std::string source, int input_size, bool print) : 
	client(client), model_id_(model_id), source_(source), input_size_(input_size), print(print)
{
}

int ModelImpl::id() { return model_id_; }

int ModelImpl::input_size() { return input_size_; }

std::string ModelImpl::source() { return source_; }

std::vector<uint8_t> ModelImpl::infer(std::vector<uint8_t> &input)
{
	auto future = infer_async(input);
	return future.get();
}

std::future<std::vector<uint8_t>> ModelImpl::infer_async(std::vector<uint8_t> &input)
{
	CHECK(input_size_ == input.size()) << "Infer called with incorrect input size";

	auto promise = std::make_shared<std::promise<std::vector<uint8_t>>>();

	auto input_data = new uint8_t[input.size()];
	std::memcpy(input_data, input.data(), input.size());

	clientapi::InferenceRequest request;
	request.header.user_id = client->user_id;
	request.header.user_request_id = client->request_id_seed++;
	request.model_id = model_id_;
	request.batch_size = 1; // TODO: support batched requests in client
	request.input_size = input.size();
	request.input = input_data;

	if (print) std::cout << "<--  " << request.str() << std::endl;

	client->connection->infer(request, [this, promise, input_data](clientapi::InferenceResponse &response) {
		if (print) std::cout << " --> " << response.str() << std::endl;
		if (response.header.status == clockworkSuccess)
		{
			uint8_t *output = static_cast<uint8_t *>(response.output);
			promise->set_value(std::vector<uint8_t>(output, output + response.output_size));
		}
		else if (response.header.status == clockworkInitializing)
		{
			promise->set_exception(std::make_exception_ptr(clockwork_initializing(response.header.message)));
		} 
		else
		{
			promise->set_exception(std::make_exception_ptr(clockwork_error(response.header.message)));
		}
		delete input_data;
		free(response.output);
	});

	return promise->get_future();
}

void ModelImpl::evict()
{
	auto future = evict_async();
	future.get();
}

std::future<void> ModelImpl::evict_async()
{
	auto promise = std::make_shared<std::promise<void>>();
	promise->set_exception(std::make_exception_ptr(clockwork_error("evict not implemented")));
	return promise->get_future();
}

Client *Connect(const std::string &hostname, const std::string &port, bool print)
{
	// ConnectionManager internally has a thread for doing IO.
	// For now just have a separate thread per client
	// Ideally clients would share threads, but for now that's nitpicking
	network::client::ConnectionManager *manager = new network::client::ConnectionManager();

	// Connect to clockwork
	network::client::Connection *clockwork_connection = manager->connect(hostname, port);

	return new NetworkClient(manager, clockwork_connection, print);
}

} // namespace clockwork