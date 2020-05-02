#include "clockwork/network/client.h"
#include "clockwork/api/client_api.h"
#include "clockwork/client.h"
#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include "clockwork/test/util.h"
#include <catch2/catch.hpp>
#include <nvml.h>
#include <iostream>
#include "clockwork/util.h"

using namespace clockwork;

uint64_t now()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string get_clockwork_dir()
{
	int bufsize = 1024;
	char buf[bufsize];
	int len = readlink("/proc/self/exe", buf, bufsize);
	return dirname(dirname(buf));
}

std::string get_example_model(std::string name = "resnet18_tesla-m40_batchsize1")
{
	return get_clockwork_dir() + "/resources/" + name + "/model";
}

class ClosedLoopClient
{
public:
	std::atomic_int request_id_seed;
	network::client::Connection *client;

	ClosedLoopClient(network::client::Connection *client) : client(client), request_id_seed(0)
	{
		loadModel();
	}

	void loadModel()
	{
		clientapi::LoadModelFromRemoteDiskRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.remote_path = get_example_model();

		std::cout << "<--  " << request.str() << std::endl;

		client->loadRemoteModel(request, [this](clientapi::LoadModelFromRemoteDiskResponse &response) {
			std::cout << " --> " << response.str() << std::endl;
			if (response.header.status == clockworkSuccess)
			{
				this->infer(response.model_id, response.input_size);
			}
		});
	}

	void infer(int model_id, int input_size)
	{
		clientapi::InferenceRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.model_id = model_id;
		request.batch_size = 1;
		request.input_size = input_size;
		request.input = malloc(input_size);

		std::cout << "<--  " << request.str() << std::endl;

		client->infer(request, [this, model_id, input_size](clientapi::InferenceResponse &response) {
			std::cout << " --> " << response.str() << std::endl;
			if (response.header.status == clockworkSuccess)
			{
				this->infer(model_id, input_size);
			}
		});
	}
};

class OpenLoopClient
{
public:
	std::default_random_engine generator;
	uint32_t rate;
	uint64_t next_arrival;
	int model_id;
	int input_size;
	std::poisson_distribution<uint64_t> p_distribution;
	std::atomic_int request_id_seed;

	network::client::Connection *client;

	OpenLoopClient(network::client::Connection *client, uint32_t _rate) : client(client), request_id_seed(0)
	{
		generator.seed(time(0));
		std::cout << "rate: " << _rate << std::endl;
		p_distribution = std::poisson_distribution<uint64_t>(pow(10, 9) / _rate); // 10^9 because the rate is inf/sec but we use nanosleep
		model_id = -1;
		input_size = -1;
		loadModel();
		sleep(5);
		while (true)
		{
			gap();
			if (model_id * input_size != 1)
			{
				this->infer(model_id, input_size);
			}
		}
	}

	void gap()
	{
		std::this_thread::sleep_for(std::chrono::nanoseconds(p_distribution(OpenLoopClient::generator)));
	}

	void loadModel()
	{
		clientapi::LoadModelFromRemoteDiskRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.remote_path = get_example_model();

		std::cout << "<--  " << request.str() << std::endl;

		client->loadRemoteModel(request, [this](clientapi::LoadModelFromRemoteDiskResponse &response) {
			std::cout << " --> " << response.str() << std::endl;
			if (response.header.status == clockworkSuccess)
			{
				model_id = response.model_id;
				input_size = response.input_size;
				std::cout << "model loaded successfully!" << std::endl;
			}
		});
	}

	void infer(int model_id, int input_size)
	{
		clientapi::InferenceRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.model_id = model_id;
		request.batch_size = 1;
		request.input_size = input_size;
		request.input = malloc(input_size);

		std::cout << "<--  " << request.str() << std::endl;

		client->infer(request, [this, model_id, input_size](clientapi::InferenceResponse &response) {
			std::cout << " --> " << response.str() << std::endl;
			if (response.header.status == clockworkSuccess)
			{
				// std::cout << " --> SUCCESS!" << std::endl;
				// 	this->infer(model_id, input_size);
			}
		});
	}
};

class BurstyClient
{
public:
	std::default_random_engine generator;
	uint32_t base_rate;
	double burst_factor;
	uint64_t burst_length;
	uint64_t burst_gap;
	uint64_t next_arrival;
	uint64_t burst_point;
	int model_id;
	int input_size;
	bool burst_mode;
	std::poisson_distribution<uint64_t> p_distribution;
	std::atomic_int request_id_seed;

	network::client::Connection *client;

	BurstyClient(network::client::Connection *client, uint32_t _rate, double _burst_factor, uint64_t _burst_length, uint64_t _burst_gap) : client(client), request_id_seed(0)
	{
		generator.seed(time(0));
		base_rate = _rate;
		burst_factor = _burst_factor;
		burst_length = _burst_length * pow(10, 9);
		burst_gap = _burst_gap * pow(10, 9);
		burst_point = burst_gap;
		p_distribution = std::poisson_distribution<uint64_t>(pow(10, 9) / base_rate); // 10^9 because the rate is inf/sec but we use nanosleep
		model_id = -1;
		input_size = -1;
		burst_mode = false;
		loadModel();
		sleep(5);
		while (true)
		{
			gap();
			if (model_id * input_size != 1)
			{
				this->infer(model_id, input_size);
			}
		}
	}

	void gap()
	{
		if (!burst_mode && (now() % (burst_gap + burst_length)) > burst_gap)
		{
			burst_mode = true;
			p_distribution = std::poisson_distribution<uint64_t>(pow(10, 9) / (base_rate * burst_factor));
		}
		else if (burst_mode && (now() % (burst_gap + burst_length)) <= burst_gap)
		{
			burst_mode = false;
			p_distribution = std::poisson_distribution<uint64_t>(pow(10, 9) / base_rate);
		}
		std::this_thread::sleep_for(std::chrono::nanoseconds(p_distribution(BurstyClient::generator)));
	}

	void loadModel()
	{
		clientapi::LoadModelFromRemoteDiskRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.remote_path = get_example_model();

		std::cout << "<--  " << request.str() << std::endl;

		client->loadRemoteModel(request, [this](clientapi::LoadModelFromRemoteDiskResponse &response) {
			std::cout << " --> " << response.str() << std::endl;
			if (response.header.status == clockworkSuccess)
			{
				model_id = response.model_id;
				input_size = response.input_size;
				std::cout << "model loaded successfully!" << std::endl;
			}
		});
	}

	void infer(int model_id, int input_size)
	{
		clientapi::InferenceRequest request;
		request.header.user_id = 0;
		request.header.user_request_id = request_id_seed++;
		request.model_id = model_id;
		request.batch_size = 1;
		request.input_size = input_size;
		request.input = malloc(input_size);

		std::cout << "<--  " << request.str() << std::endl;

		client->infer(request, [this, model_id, input_size](clientapi::InferenceResponse &response) {
			std::cout << " --> " << response.str() << std::endl;
			if (response.header.status == clockworkSuccess)
			{
				// std::cout << " --> SUCCESS!" << std::endl;
				// 	this->infer(model_id, input_size);
			}
		});
	}
};

int main(int argc, char *argv[])
{
	// model_id, workload_type, rate, burst_factor, burst_length, burst_gap

	if (argc != 3)
	{
		std::cerr << "Usage: controller HOST PORT" << std::endl;
		return 1;
	}
	std::cout << "Starting Clockwork Client" << std::endl;

	clockwork::Client *client = clockwork::Connect(argv[1], argv[2]);

	clockwork::Model *model = client->load_remote_model(get_example_model());

	// OpenLoopClient *open_loop = new OpenLoopClient(client, 100);

	while (true)
	{
		std::vector<uint8_t> input(model->input_size());
		model->infer(input);
		usleep(1000000);
	}

	std::cout << "Clockwork Client Exiting" << std::endl;
}