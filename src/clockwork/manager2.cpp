#include "clockwork/manager2.h"


void ParamsEvictionHandler::evicted() {
	model->params_evicted();
}

void WorkspaceEvictionHandler::evicted() {
	model->workspace_evicted();
}