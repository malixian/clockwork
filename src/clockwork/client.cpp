#include <cstring>
#include <future>
#include <atomic>
#include <dmlc/logging.h>
#include "clockwork/client.h"
#include "clockwork/api/client_api.h"
#include "clockwork/network/client.h"
#include "clockwork/telemetry/client_telemetry_logger.h"

namespace clockwork
{

class NetworkClient : public Client
{
public:
	bool print;
	ClientTelemetryLogger* telemetry;
	std::atomic_int request_id_seed;
	int user_id;
	network::client::ConnectionManager *manager;
	network::client::Connection *connection;
	ModelSet models;

	NetworkClient(network::client::ConnectionManager *manager, network::client::Connection *connection, bool print, bool summarize);
	virtual ~NetworkClient();

	virtual Model *get_model(int model_id);
	virtual std::future<Model *> get_model_async(int model_id);
	virtual Model *upload_model(std::vector<uint8_t> &serialized_model);
	virtual std::future<Model *> upload_model_async(std::vector<uint8_t> &serialized_model);
	virtual Model* load_remote_model(std::string model_path);
	virtual std::future<Model *> load_remote_model_async(std::string model_path);
	virtual std::vector<Model*> load_remote_models(std::string model_path, int no_of_copies);
	virtual std::future<std::vector<Model *>> load_remote_models_async(std::string model_path, int no_of_copies);
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
	const size_t input_size_;
	const size_t output_size_;

	ModelImpl(NetworkClient *client, int model_id, std::string source, size_t input_size_, size_t output_size, bool print);

	virtual int id();
	virtual size_t input_size();
	virtual size_t output_size();
	virtual std::string source();

	virtual std::vector<uint8_t> infer(std::vector<uint8_t> &input);
	virtual std::future<std::vector<uint8_t>> infer_async(std::vector<uint8_t> &input);
	virtual void evict();
	virtual std::future<void> evict_async();
};

NetworkClient::NetworkClient(network::client::ConnectionManager *manager, 
	network::client::Connection *connection, bool print, bool summarize) : 
	manager(manager), connection(connection), user_id(0), request_id_seed(0), 
	print(print)
{
	if (summarize) {
		telemetry = new ClientTelemetrySummarizer();
	} else {
		telemetry = new NoOpClientTelemetryLogger();
	}
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

Model* NetworkClient::load_remote_model(std::string model_path) {
	return load_remote_model_async(model_path).get();
}

std::future<Model *> NetworkClient::load_remote_model_async(std::string model_path) {
	auto promise = std::make_shared<std::promise<Model *>>();
	promise->set_value(load_remote_models_async(model_path, 1).get()[0]);
	return promise->get_future();
}

std::vector<Model*> NetworkClient::load_remote_models(std::string model_path, int no_of_copies)
{
	return load_remote_models_async(model_path, no_of_copies).get();
}

std::future<std::vector<Model *>> NetworkClient::load_remote_models_async(std::string model_path, int no_of_copies)
{
	auto promise = std::make_shared<std::promise<std::vector<Model *>>>();

	clientapi::LoadModelFromRemoteDiskRequest load_model;
	load_model.header.user_id = 0;
	load_model.header.user_request_id = request_id_seed++;
	load_model.remote_path = model_path;
	load_model.no_of_copies = no_of_copies;

	if (print) std::cout << "<--  " << load_model.str() << std::endl;

	connection->loadRemoteModel(load_model, [this, promise, model_path](clientapi::LoadModelFromRemoteDiskResponse &response) {
		if (print) std::cout << " --> " << response.str() << std::endl;
		if (response.header.status == clockworkSuccess)
		{
			std::vector<Model*> models = std::vector<Model*>();
			for (int i = 0; i < response.copies_created; i++)
				models.push_back(new ModelImpl(this, (response.model_id + i), model_path, response.input_size, response.output_size, print));
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
				auto impl = new ModelImpl(this, model.model_id, model.remote_path, model.input_size, model.output_size, print);
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

ModelImpl::ModelImpl(NetworkClient *client, int model_id, std::string source, size_t input_size, size_t output_size, bool print) : 
	client(client), model_id_(model_id), source_(source), input_size_(input_size), output_size_(output_size), print(print)
{
}

int ModelImpl::id() { return model_id_; }

size_t ModelImpl::input_size() { return input_size_; }
size_t ModelImpl::output_size() { return output_size_; }

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

	uint64_t t_send = util::now();
	client->connection->infer(request, [this, promise, input_data, t_send](clientapi::InferenceResponse &response) {
		uint64_t t_receive = util::now();
		if (print) std::cout << " --> " << response.str() << std::endl;
		if (response.header.status == clockworkSuccess)
		{
			uint8_t *output = static_cast<uint8_t *>(response.output);
			promise->set_value(std::vector<uint8_t>(output, output + response.output_size));
			client->telemetry->log(client->user_id, model_id_, 1, input_size_, output_size_, t_send, t_receive, true);
		}
		else if (response.header.status == clockworkInitializing)
		{
			promise->set_exception(std::make_exception_ptr(clockwork_initializing(response.header.message)));
		} 
		else
		{
			promise->set_exception(std::make_exception_ptr(clockwork_error(response.header.message)));
			client->telemetry->log(client->user_id, model_id_, 1, input_size_, output_size_, t_send, t_receive, false);
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

Client *Connect(const std::string &hostname, const std::string &port, bool print, bool summarize)
{
	// ConnectionManager internally has a thread for doing IO.
	// For now just have a separate thread per client
	// Ideally clients would share threads, but for now that's nitpicking
	network::client::ConnectionManager *manager = new network::client::ConnectionManager();

	// Connect to clockwork
	network::client::Connection *clockwork_connection = manager->connect(hostname, port);

	return new NetworkClient(manager, clockwork_connection, print, summarize);
}

} // namespace clockwork
