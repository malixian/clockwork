#ifndef _CLOCKWORK_WORKLOAD_SLO_H_
#define _CLOCKWORK_WORKLOAD_SLO_H_

#include "clockwork/util.h"
#include "clockwork/workload/workload.h"
#include "clockwork/workload/azure.h"
#include <cstdlib>
#include <iostream>

namespace clockwork {
namespace workload {

Engine* slo_experiment_1(
	clockwork::Client* client,
	std::string model_name,
	unsigned num_copies,
	std::string distribution,
	double rate,
	unsigned slo_start,
	unsigned slo_end,
	double slo_factor,
	std::string slo_op,
	unsigned period) {

	Engine* engine = new Engine();

	std::string modelpath;
	try {
		modelpath = util::get_clockwork_modelzoo()[model_name];
	} catch (const std::exception& e) {
		CHECK(false) << "Modelpath not found: " << e.what();
	}

	auto models = client->load_remote_models(modelpath, num_copies);

	for (unsigned i = 0; i < models.size(); i++) {
		if (distribution == "poisson") {
			engine->AddWorkload(new PoissonOpenLoop(
				i,				// client id, just use the same for this
				models[i],		// model
				i,				// rng seed
				rate/num_copies // requests/second
			));
		} else if (distribution == "fixed-rate") {
			engine->AddWorkload(new FixedRate(
				i,				// client id, just use the same for this
				models[i],		// model
				i,				// rng seed
				rate/num_copies // requests/second
			));
		} else {
			CHECK(false) << distribution << " distribution not yet implemented";
		}
	}

	if (slo_op == "add") {
		// adjust all models additively
		engine->AddWorkload(new AdjustSLO(
			period, 	// period in seconds
			slo_start,	// initial slo
			models,		// apply to all models, for now
			[slo_factor](float current) { return current + slo_factor; },	// slo update function
			[slo_end](float current) { return current > slo_end; } 	// slo termination condition
		));
	} else if (slo_op == "mul") {
		engine->AddWorkload(new AdjustSLO(
			period, 	// period in seconds
			slo_start,	// initial slo
			models,		// apply to all models, for now
			[slo_factor](float current) { return current * slo_factor; },	// slo update function
			[slo_end](float current) { return current > slo_end; } 	// slo termination condition
		));
	} else {
		CHECK(false) << slo_op << " operator not yet implemented";
	}

	return engine;	
}

}
}

#endif 
