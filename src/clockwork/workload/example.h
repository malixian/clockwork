#ifndef _CLOCKWORK_WORKLOAD_EXAMPLE_H_
#define _CLOCKWORK_WORKLOAD_EXAMPLE_H_

#include "clockwork/workload/workload.h"
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
		engine->AddWorkload(new ClosedLoop(
			i, 			// client id
			models[i],	// model
			1			// concurrency
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

}
}

#endif 

