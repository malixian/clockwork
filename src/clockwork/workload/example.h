#ifndef _CLOCKWORK_WORKLOAD_EXAMPLE_H_
#define _CLOCKWORK_WORKLOAD_EXAMPLE_H_

#include "clockwork/util.h"
#include "clockwork/workload/workload.h"
#include "clockwork/workload/azure.h"
#include <cstdlib>
#include <iostream>

namespace clockwork {
namespace workload {

Engine* simple(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 2;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		models[i]->disable_inputs();
		engine->AddWorkload(new ClosedLoop(
			0, 			// client id, just use the same for this
			models[i],	// model
			1			// concurrency
		));
	}

	return engine;
}

Engine* simple_slo_factor(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 3;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new ClosedLoop(
			0, 			// client id, just use the same for this
			models[i],	// model
			1			// concurrency
		));
	}

	// Adjust model 0 multiplicatively
	engine->AddWorkload(new AdjustSLO(
		10.0, 					// period in seconds
		4.0,					// initial slo_factor
		{models[0]},			// The model or models to apply the SLO adjustment to
		[](float current) { 	// SLO update function
			return current * 1.25; 
		},
		[](float current) { return false; } // Terminate condition
	));

	// Adjust model 1 additively
	engine->AddWorkload(new AdjustSLO(
		10.0, 					// period in seconds
		4.0,					// initial slo_factor
		{models[1]},			// The model or models to apply the SLO adjustment to
		[](float current) { 	// SLO update function
			return current + 1.0; 
		},
		[](float current) { return false; } // Terminate condition
	));

	// Adjust model 2 back and forth
	engine->AddWorkload(new AdjustSLO(
		10.0, 					// period in seconds
		10,						// initial slo_factor
		{models[2]},			// The model or models to apply the SLO adjustment to
		[](float current) { 	// SLO update function
			return current = 10 ? 1 : 10;
		},
		[](float current) { return false; } // Terminate condition
	));

	return engine;	
}

Engine* simple_parametric(clockwork::Client* client, unsigned num_copies,
	unsigned concurrency, unsigned num_requests) {
	Engine* engine = new Engine();

	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		std::cout << "Adding a ClosedLoop Engine for Model " << i << std::endl;
		engine->AddWorkload(new ClosedLoop(
			i, 				// client id
			models[i],		// model
			concurrency,	// concurrency
			num_requests,	// max num requests
			0				// jitter
		));
	}

	return engine;
}

Engine* poisson_open_loop(clockwork::Client* client, unsigned num_models,
	double rate) {
	Engine* engine = new Engine();

	std::string model_name = "resnet50_v2";
	std::string modelpath = util::get_clockwork_modelzoo()[model_name];

	std::cout << "Loading " << num_models << " " << model_name
			  << " models" << std::endl;
	std::cout << "Cumulative request rate across models: " << rate
			  << " requests/seconds" << std::endl;
	auto models = client->load_remote_models(modelpath, num_models);

	std::cout << "Adding a PoissonOpenLoop Workload (" << (rate/num_models)
			  << " requests/second) for each model" << std::endl;
	for (unsigned i = 0; i < models.size(); i++) {
		engine->AddWorkload(new PoissonOpenLoop(
			i,				// client id
			models[i],  	// model
			i,      		// rng seed
			rate/num_models	// requests/second
		));
	}

	return engine;
}

Engine* example(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned client_id = 0;
	for (auto &p : util::get_clockwork_modelzoo()) {
		std::cout << "Loading " << p.first << std::endl;
		auto model = client->load_remote_model(p.second);

		Workload* fixed_rate = new FixedRate(
			client_id,		// client id
			model,  		// model
			0,      		// rng seed
			5				// requests/second
		);
		Workload* open = new PoissonOpenLoop(
			client_id,		// client id
			model,			// model
			1,				// rng seed
			10				// request/second
		);
		Workload* burstyopen = new BurstyPoissonOpenLoop(
			client_id, 		// client id
			model, 			// model
			1,				// rng seed
			10,				// requests/second
			10,				// burst duration
			20				// idle duration
		);
		Workload* closed = new ClosedLoop(
			client_id, 		// client id
			model,			// model
			1				// concurrency
		);
		Workload* burstyclosed = new BurstyPoissonClosedLoop(
			client_id, 		// client id
			model, 			// model
			1,				// concurrency
			0,				// rng seed
			10, 			// burst duration
			20				// idle duration
		);

		engine->AddWorkload(fixed_rate);
		engine->AddWorkload(open);
		engine->AddWorkload(burstyopen);
		engine->AddWorkload(closed);
		engine->AddWorkload(burstyclosed);

		client_id++;
	}

	return engine;
}

Engine* spam(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 100;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		models[i]->disable_inputs();
		engine->AddWorkload(new ClosedLoop(
			0, 			// client id, just use the same for this
			models[i],	// model
			100			// concurrency
		));
	}

	return engine;
}

Engine* single_spam(clockwork::Client* client) {
	Engine* engine = new Engine();

	unsigned num_copies = 1;
	std::string modelpath = util::get_clockwork_modelzoo()["resnet50_v2"];
	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		models[i]->disable_inputs();
		engine->AddWorkload(new ClosedLoop(
			0, 			// client id, just use the same for this
			models[i],	// model
			1000			// concurrency
		));
	}

	return engine;
}

Engine* azure(clockwork::Client* client) {
	Engine* engine = new Engine();

	auto trace_data = azure::load_trace();

	unsigned trace_id = 1;
	std::string model = util::get_clockwork_model("resnet50_v2");
	unsigned num_copies = 200;

	auto models = client->load_remote_models(model, num_copies);

	for (unsigned i = 0; i < trace_data.size(); i++) {
		auto model = models[i % num_copies];
		auto workload = trace_data[i];

		Workload* replay = new PoissonTraceReplay(
			0,				// client id, just give them all the same ID for this example
			model,			// model
			i,				// rng seed
			workload,		// trace data
			1.0,			// scale factor; default 1
			60.0,			// interval duration; default 60
			0				// interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
	}

	return engine;
}

Engine* azure_small(clockwork::Client* client) {
	Engine* engine = new Engine();

	auto trace_data = azure::load_trace();

	std::vector<Model*> models;
	for (auto &p : util::get_clockwork_modelzoo()) {
		std::cout << "Loading " << p.first << std::endl;
		for (auto &model : client->load_remote_models(p.second, 3)) {
			models.push_back(model);
		}
	}

	for (unsigned i = 0; i < trace_data.size(); i++) {
		auto model = models[i % models.size()];
		model->disable_inputs();
		auto workload = trace_data[i];

		Workload* replay = new PoissonTraceReplay(
			i,				// client id, just give them all the same ID for this example
			model,			// model
			i,				// rng seed
			workload,		// trace data
			1.0,			// scale factor; default 1
			60.0,			// interval duration; default 60
			0				// interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
	}

	return engine;
}

Engine* azure_fast(clockwork::Client* client, unsigned trace_id = 1) {
	Engine* engine = new Engine();

	auto trace_data = azure::load_trace(trace_id);

	std::vector<Model*> models;
	for (auto &p : util::get_clockwork_modelzoo()) {
		std::cout << "Loading " << p.first << std::endl;
		for (auto &model : client->load_remote_models(p.second, 3)) {
			models.push_back(model);
		}
	}


	for (unsigned i = 0; i < trace_data.size(); i++) {
		auto model = models[i % models.size()];
		// model->disable_inputs();
		auto workload = trace_data[i];

		Workload* replay = new PoissonTraceReplay(
			i,				// client id, just give them all the same ID for this example
			model,			// model
			i,				// rng seed
			workload,		// trace data
			1.0,			// scale factor; default 1
			1.0,			// interval duration; default 60
			-1				// interval to begin at; default 0; set to -1 for random
		);

		engine->AddWorkload(replay);
	}

	return engine;
}

}
}

#endif 

