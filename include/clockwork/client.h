#ifndef _CLOCKWORK_CLIENT_H_
#define _CLOCKWORK_CLIENT_H_

#include <functional>
#include <string>

/**
This is the user-facing Clockwork Client
*/

namespace clockwork {

/** A super simple representation of a model -- blobs of data */
class ClockworkModel {
public:
	std::string so, params, clockwork;

	static ClockworkModel* LoadFromLocalDisk(std::string &so_filename, std::string &params_filename, std::string &clockwork_filename);
};

class Client;
class LocalClient;
class RemoteClient;

class Client {
public:

	/** The proper way of uploading a model will be to send it an ONNX file,
	where it will be compiled remotely.  For now we'll pre-compile clockwork
	models.  This is the synchronous version.  On error, will throw an exception. */
	void uploadModel(ClockworkModel* model, int &model_id);

	/** Asynchronous version of uploadModel.
	On success, callback will be called with the model id.
	On error, errback will be called with the status code and the error message */
	void uploadModelAsync(ClockworkModel* model, std::function<void(int)> callback, std::function<void(int, std::string)> errback);

	/** From the client's perspective this is the only API call for inference */
	void infer(int model_id, int inputSize, void* input, int* outputSize, void** output);

	/** Asynchronous version of infer.
	On success, callback will be called with the output.
	On error, errback will be called with the status code and the error message */
	void inferAsync(int model_id, int inputSize, void* input, std::function<void(int, void*)> callback, std::function<void(int, std::string)> errback);

	/** Connect to a clockwork instance over the network */
	static RemoteClient* ConnectTo(std::string hostname, int port);

	/** Connect to a clockwork instance running on the local machine */
	static LocalClient* Local(); // TODO TODO this
};

class LocalClient : public Client{};
class RemoteClient : public Client{};

}

#endif