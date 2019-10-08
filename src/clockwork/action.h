#ifndef _CLOCKWORK_ACTION_H_
#define _CLOCKWORK_ACTION_H_

#include <thread>
#include <limits>
#include <algorithm>
#include <memory>
#include <atomic>
#include <cuda_runtime.h>
#include "tvm/runtime/cuda_common.h"
#include "clockwork/telemetry.h"
#include "clockwork/cache.h"
#include "clockwork/model/model.h"
#include "clockwork/priority_queue.h"
#include "clockwork/common.h"
#include "tbb/concurrent_queue.h"
#include "clockwork/task.h"
#include "clockwork/runtime.h"

namespace clockwork {

class Action {
public:
	virtual void submit() = 0;
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;
};

class LoadWeightsAction : public Action {
private:

	class LoadWeightsTaskImpl : public LoadWeightsTask {
	public:
		LoadWeightsAction* action;

		LoadWeightsTaskImpl(LoadWeightsAction* action);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm);
		void error(int status_code, std::string message);
	};

	ClockworkRuntime* runtime;
	int model_id;
	uint64_t earliest, latest;
	LoadWeightsTaskImpl* task;

public:
	LoadWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest);
	~LoadWeightsAction();
	void submit();
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;
};

class EvictWeightsAction : public Action {
private:

	class EvictWeightsTaskImpl : public EvictWeightsTask {
	public:
		EvictWeightsAction* action;

		EvictWeightsTaskImpl(EvictWeightsAction* action);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm);
		void error(int status_code, std::string message);
	};

	ClockworkRuntime* runtime;
	int model_id;
	uint64_t earliest, latest;
	EvictWeightsTaskImpl* task;

public:
	EvictWeightsAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest);
	~EvictWeightsAction();
	void submit();
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;
};


class InferAction : public Action {
private:

	class CopyOutputTaskImpl : public CopyOutputTask {
	public:
		InferAction* action;

		CopyOutputTaskImpl(InferAction* action);

		void run(cudaStream_t stream);
		void success();
		void error(int status_code, std::string message);
	};


	class InferTaskImpl : public InferTask {
	public:
		InferAction* action;

		InferTaskImpl(InferAction* action);

		void run(cudaStream_t stream);
		void success();
		void error(int status_code, std::string message);
	};


	class CopyInputTaskImpl : public CopyInputTask {
	public:
		InferAction* action;

		CopyInputTaskImpl(InferAction* action);

		void run(cudaStream_t stream);
		void success(RuntimeModel* rm, std::shared_ptr<Allocation> workspace);
		void error(int status_code, std::string message);
	};

	ClockworkRuntime* runtime;

	int model_id;
	uint64_t earliest, latest;
	char* input;
	char* output;

	RuntimeModel* rm;
	std::shared_ptr<Allocation> workspace;

	CopyInputTaskImpl* copy_input = nullptr;
	InferTaskImpl* infer = nullptr;
	CopyOutputTaskImpl* copy_output = nullptr;
	

	uint64_t copy_input_earliest();

public:
	InferAction(ClockworkRuntime* runtime, int model_id, uint64_t earliest, uint64_t latest, char* input, char* output);
	~InferAction();

	void submit();
	virtual void success() = 0;
	virtual void error(int status_code, std::string message) = 0;

};



}

#endif
