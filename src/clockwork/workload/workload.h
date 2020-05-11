#ifndef _CLOCKWORK_WORKLOAD_WORKLOAD_H_
#define _CLOCKWORK_WORKLOAD_WORKLOAD_H_

#include <queue>
#include <cstdint>
#include <functional>
#include <vector>
#include "clockwork/client.h"
#include "tbb/concurrent_queue.h"
#include <random>

namespace clockwork {
namespace workload {

class Workload;

class Engine {
private:
	struct element {
		uint64_t ready;
		std::function<void(void)> callback;

		friend bool operator < (const element& lhs, const element &rhs) {
			return lhs.ready < rhs.ready;
		}
		friend bool operator > (const element& lhs, const element &rhs) {
			return lhs.ready > rhs.ready;
		}
	};

	uint64_t now;
	tbb::concurrent_queue<std::function<void(void)>> runqueue;
	std::priority_queue<element, std::vector<element>, std::greater<element>> queue;
	std::vector<Workload*> workloads;

public:

	void AddWorkload(Workload* workload);
	void SetTimeout(uint64_t timeout, std::function<void(void)> callback);
	void InferComplete(Workload* workload, unsigned model_index);
	void InferError(Workload* workload, unsigned model_index, int status);

	void Run(clockwork::Client* client);

};

class Workload {
private:
	std::vector<clockwork::Model*> models;
	Engine* engine;

public:
	int user_id;

	Workload(int id);
	Workload(int id, clockwork::Model* model);
	Workload(int id, std::vector<clockwork::Model*> &models);

	// Used during workload initialization
	void AddModel(clockwork::Model* model);
	void AddModels(std::vector<clockwork::Model*> &models);
	void SetEngine(Engine* engine);

	// Methods to be called by subclasses
	void Infer(unsigned model_index = 0);
	void SetTimeout(uint64_t timeout, std::function<void(void)> callback);

	// Methods to be implemented by subclasses
	virtual void Start(uint64_t now) = 0;
	virtual void InferComplete(uint64_t now, unsigned model_index) = 0;
	virtual void InferError(uint64_t now, unsigned model_index, int status) = 0;

	// Optional methods to be overridden by subclasses
	virtual void InferErrorInitializing(uint64_t now, unsigned model_index);

};

class ClosedLoop : public Workload {
public:
	const unsigned concurrency;

	ClosedLoop(int id, clockwork::Model* model, unsigned concurrency);

	virtual void Start(uint64_t now);
	virtual void InferComplete(uint64_t now, unsigned model_index);
	virtual void InferError(uint64_t now, unsigned model_index, int status);
	virtual void InferErrorInitializing(uint64_t now, unsigned model_index);
};

template <typename TDISTRIBUTION> class OpenLoop : public Workload {
public:
	std::minstd_rand rng;
	TDISTRIBUTION distribution;

	OpenLoop(int id, clockwork::Model* model, int rng_seed, TDISTRIBUTION distribution) : 
		Workload(id, model), rng(rng_seed), distribution(distribution) {
	}

	void Submit() {
		Infer(0);
		uint64_t timeout = distribution(rng);
		SetTimeout(timeout, [this]() { Submit(); });
	}

	void Start(uint64_t now) {
		SetTimeout(distribution(rng), [this]() { Submit(); });
	}

	void InferComplete(uint64_t now, unsigned model_index) {}
	void InferError(uint64_t now, unsigned model_index, int status) {}

};

class Static {
public:
	const uint64_t value;
	Static(uint64_t value) : value(value) {}
	const uint64_t& operator()(std::minstd_rand rng) const { return value; }
};

typedef std::poisson_distribution<uint64_t> Poisson;

class FixedRate : public OpenLoop<Static> {
public:
	// mean is provided in seconds
	FixedRate(int id, clockwork::Model* model, int rng_seed, double rate) : 
		OpenLoop(id, model, rng_seed, Static(1000000000.0 / rate)) {
	}

};

class PoissonOpenLoop : public OpenLoop<Poisson> {
public:
	// mean is provided in seconds
	PoissonOpenLoop(int id, clockwork::Model* model, int rng_seed, double rate) : 
		OpenLoop(id, model, rng_seed, Poisson(1000000000.0 / rate)) {
	}

};

template <typename DBURST, typename DIDLE> 
class BurstyClosedLoop : public Workload {
public:
	unsigned concurrency;
	std::minstd_rand rng;
	DBURST d_burst;
	DIDLE d_idle;

	bool bursting = false;
	int outstanding = 0;

	BurstyClosedLoop(int id, 
		clockwork::Model* model,
		unsigned concurrency,
		int rng_seed, 
		DBURST d_burst,
		DIDLE d_idle) : 
			Workload(id, model), 
			concurrency(concurrency),
			rng(rng_seed),
			d_burst(d_burst),
			d_idle(d_idle)
	{}

	void StartBursting(uint64_t duration) {
		bursting = true;
		for (unsigned i = 0; i < concurrency; i++) {
			outstanding++;
			Infer();
		}
		SetTimeout(duration, [this]() { bursting = false; });
	}

	void ScheduleNextBurst(uint64_t interval) {
		SetTimeout(interval, [this]() { 
			StartBursting(d_burst(rng)); 
		});
	}

	void Start(uint64_t now) {
		double s = static_cast<double>(rng()) / static_cast<double>(UINT64_MAX);
		uint64_t burst = d_burst(rng);
		uint64_t idle = d_idle(rng);
		uint64_t initial_period = burst + idle;
		uint64_t start_at = static_cast<uint64_t>(((double)initial_period) * s);
		if (start_at > burst) {
			ScheduleNextBurst(initial_period - start_at);
		} else {
			StartBursting(burst - start_at);
		}
	}

	void RequestComplete() {
		outstanding--;
		if (bursting) {
			Infer();
			outstanding++;
		} else if (outstanding == 0) {
			ScheduleNextBurst(d_burst(rng));
		}
	}

	void InferComplete(uint64_t now, unsigned model_index) { RequestComplete(); }
	void InferError(uint64_t now, unsigned model_index, int status) { RequestComplete(); }
	void InferErrorInitializing(uint64_t now, unsigned model_index) { RequestComplete(); }

};


class BurstyPoissonClosedLoop : public BurstyClosedLoop<Poisson, Poisson> {
public:
	// mean is provided in seconds
	BurstyPoissonClosedLoop(int id, clockwork::Model* model, unsigned concurrency,
		int rng_seed, double burstIntervalSeconds, double burstDurationSeconds) : 
		BurstyClosedLoop(id, model, concurrency, rng_seed,
			Poisson(1000000000.0 * burstIntervalSeconds),
			Poisson(1000000000.0 * burstDurationSeconds)) {
	}

};

template <typename DREQUEST, typename DBURST, typename DIDLE> 
class BurstyOpenLoop : public Workload {
public:
	int current_burst = 0;
	std::minstd_rand rng_request;
	std::minstd_rand rng_burst;
	DREQUEST d_request;
	DBURST d_burst;
	DIDLE d_idle;

	BurstyOpenLoop(int id, 
		clockwork::Model* model, 
		int rng_seed, 
		DREQUEST d_request,
		DBURST d_burst,
		DIDLE d_idle) : 
			Workload(id, model), 
			rng_request(rng_seed+1), 
			rng_burst(rng_seed), 
			d_request(d_request),
			d_burst(d_burst),
			d_idle(d_idle)
	{}

	void Submit(int burst) {
		if (burst == this->current_burst) {
			Infer(0);
			uint64_t timeout = d_request(rng_request);
			SetTimeout(timeout, [this, burst]() { Submit(burst); });
		}
	}

	void StartBursting() {
		Submit(current_burst);
		uint64_t stop_burst_at = d_burst(rng_burst);
		SetTimeout(stop_burst_at, [this]() { StopBursting(); });
	}

	void StopBursting() {
		current_burst++;
		uint64_t next_burst_at = d_idle(rng_burst);
		SetTimeout(next_burst_at, [this]() { StartBursting(); });
	}

	void Start(uint64_t now) {
		double s = static_cast<double>(rng_burst()) / static_cast<double>(UINT64_MAX);
		uint64_t burst = d_burst(rng_burst);
		uint64_t idle = d_idle(rng_burst);
		uint64_t initial_period = burst + idle;
		uint64_t start_at = static_cast<uint64_t>(((double)initial_period) * s);
		if (start_at > burst) {
			uint64_t next_burst_at = initial_period - start_at;
			SetTimeout(next_burst_at, [this]() { StartBursting(); });
		} else {
			uint64_t stop_burst_at = burst - start_at;
			Submit(current_burst);
			SetTimeout(stop_burst_at, [this]() { StopBursting(); });
		}
	}

	void InferComplete(uint64_t now, unsigned model_index) {}
	void InferError(uint64_t now, unsigned model_index, int status) {}

};

class BurstyPoissonOpenLoop : public BurstyOpenLoop<Poisson, Poisson, Poisson> {
public:
	// mean is provided in seconds
	BurstyPoissonOpenLoop(int id, clockwork::Model* model, int rng_seed, 
			double rate, double burstDurationSeconds, double idleDurationSeconds) : 
		BurstyOpenLoop(id, model, rng_seed, 
			Poisson(1000000000.0 / rate),
			Poisson(1000000000.0 * burstDurationSeconds),
			Poisson(1000000000.0 * idleDurationSeconds)) {
	}

};

}
}

#endif 

