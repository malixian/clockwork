#ifndef _CLOCKWORK_CLIENT_H_
#define _CLOCKWORK_CLIENT_H_

#include <future>
#include <functional>
#include <string>

/**
This is the user-facing Clockwork Client
*/

namespace clockwork {

/* Represents a model that can be inferred */
class ClockworkModel {
public:
	int model_id;
	int input_size;

	/* 
	Perform an inference with the provided input and return the output.
	Blocks until the inference has completed.
	Can throw exceptions.
	*/
	virtual std::vector<uint8_t> infer(std::vector<uint8_t> &input) = 0;

	/*
	Asynchronous version of infer.
	Performans an inference on the provided input.
	Returns a future that will receive the result of the inference (the output size and the output)
	If an exception occurs, it will be thrown by calls to future.get().
	*/
	virtual std::future<std::vector<uint8_t>> infer_async(std::vector<uint8_t> &input) = 0;

	/*
	This is a backdoor API call that's useful for testing.
	Instructs the server to evict the weights of this model from the GPU.
	Will throw an exception is the weights aren't loaded.
	*/
	virtual void evict() = 0;
	virtual std::future<void> evict_async() = 0;

};

/* 
Represents a client to Clockwork 
Clockwork can be either local or remote,
and Clockwork can have multiple clients
*/
class ClockworkClient {
public:

	/*
	Gets an existing ClockworkModel from Clockwork that can then be inferenced.
	Can throw exceptions including if the model doesn't exist.
	*/
	virtual ClockworkModel* get_model(int model_id) = 0;
	virtual std::future<ClockworkModel*> get_model_async(int model_id) = 0;

	/*
	Uploads a model to Clockwork.  Returns a ClockworkModel for the model
	Can throw exceptions including if the model is invalid
	*/
	virtual ClockworkModel* upload_model(std::vector<uint8_t> &serialized_model) = 0;
	virtual std::future<ClockworkModel*> upload_model_async(std::vector<uint8_t> &serialized_model) = 0;

};

class Clockwork {
public:

	/* Connect to a Clockwork instance */
	static ClockworkClient* Connect(std::string &hostname, int port);

};

}

#endif