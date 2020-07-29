#ifndef _CLOCKWORK_TEST_ACTIONS_DUMMY_H_
#define _CLOCKWORK_TEST_ACTIONS_DUMMY_H_

#include "clockwork/api/worker_api.h"
#include "clockwork/test/actions.h"
#include "clockwork/dummy/clockwork/action_dummy.h"
#include "clockwork/dummy/clockwork/worker_dummy.h"


namespace clockwork {

class TestLoadModelFromDiskDummy : public LoadModelFromDiskDummyAction, public TestAction {
public:
    TestLoadModelFromDiskDummy(MemoryManagerDummy* Manager, std::shared_ptr<workerapi::LoadModelFromDisk> action) : 
        LoadModelFromDiskDummyAction(Manager, action) {}

    void success(std::shared_ptr<workerapi::LoadModelFromDiskResult> result) {
        setsuccess();
    }

    void error(int status_code, std::string message){
        auto result = std::make_shared<workerapi::ErrorResult>();
        result->action_type = workerapi::loadModelFromDiskAction;
        result->status = status_code; 
        seterror(result);
    }

};

class TestLoadWeightsDummy : public LoadWeightsDummyAction, public TestAction {
public:
    TestLoadWeightsDummy(MemoryManagerDummy* Manager, std::shared_ptr<workerapi::LoadWeights> action) : 
        LoadWeightsDummyAction(Manager, action) {}

    void success(std::shared_ptr<workerapi::LoadWeightsResult> result) {
        setsuccess();
    }

    void error(int status_code, std::string message) {
        auto result = std::make_shared<workerapi::ErrorResult>();
        result->action_type = workerapi::loadWeightsAction;
        result->status = status_code;
        seterror(result);
    }

};


class TestEvictWeightsDummy : public EvictWeightsDummyAction, public TestAction {
public:
    TestEvictWeightsDummy(MemoryManagerDummy* Manager, std::shared_ptr<workerapi::EvictWeights> action) : 
        EvictWeightsDummyAction(Manager, action) {}

    void success(std::shared_ptr<workerapi::EvictWeightsResult> result) {
        setsuccess();
    }

    void error(int status_code, std::string message) {
        auto result = std::make_shared<workerapi::ErrorResult>();
        result->action_type = workerapi::evictWeightsAction;
        result->status = status_code;
        seterror(result);
    }

};


class TestInferDummy : public InferDummyAction, public TestAction {
public:
	TestInferDummy(MemoryManagerDummy* Manager, std::shared_ptr<workerapi::Infer> action) : 
		InferDummyAction(Manager, action) {}

	void success(std::shared_ptr<workerapi::InferResult> result) {
		setsuccess();
	}

    void error(int status_code, std::string message) {
        auto result = std::make_shared<workerapi::ErrorResult>();
        result->action_type = workerapi::inferAction;
        result->status = status_code;
        seterror(result);
    }

};

class ClockworkRuntimeWrapperDummy : public ClockworkRuntimeDummy {
public:
    ~ClockworkRuntimeWrapperDummy() {
        this->shutdown(true);
    }
};

std::shared_ptr<workerapi::Infer> infer_action(int batch_size, BatchedModel* model);

std::shared_ptr<workerapi::Infer> infer_action2(ClockworkRuntimeDummy* worker);

}

#endif