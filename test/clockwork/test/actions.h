#ifndef _CLOCKWORK_TEST_ACTIONS_H_
#define _CLOCKWORK_TEST_ACTIONS_H_

#include "clockwork/api/worker_api.h"
#include "clockwork/util.h"
#include "clockwork/test/util.h"
#include "clockwork/model/model.h"
#include "clockwork/worker.h"
#include <memory>

using namespace clockwork::model;

namespace clockwork {

std::shared_ptr<workerapi::LoadModelFromDisk> load_model_from_disk_action();

std::shared_ptr<workerapi::LoadWeights> load_weights_action();

std::shared_ptr<workerapi::EvictWeights> evict_weights_action();

std::shared_ptr<workerapi::Infer> infer_action();

std::shared_ptr<workerapi::Infer> infer_action(int batch_size, BatchedModel* model);

std::shared_ptr<workerapi::Infer> infer_action2(ClockworkWorker* worker);

std::shared_ptr<workerapi::GetWorkerState> get_worker_state_action();

}

#endif