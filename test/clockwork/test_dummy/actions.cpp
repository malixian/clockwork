#include "clockwork/test_dummy/actions.h"

namespace clockwork {

std::shared_ptr<workerapi::Infer> infer_action(int batch_size, RuntimeModelDummy* model) {
    auto action = infer_action();
    action->batch_size = batch_size;
    action->input_size = model->input_size(batch_size);
    return action;
}

}
