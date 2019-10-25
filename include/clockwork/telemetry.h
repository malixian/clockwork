#ifndef _CLOCKWORK_TELEMETRY_H_
#define _CLOCKWORK_TELEMETRY_H_

#include <pods/pods.h>
#include <pods/binary.h>
#include <pods/buffers.h>
#include <chrono>

namespace clockwork {

struct TaskTelemetry {
	int task_type, executor_id, model_id, action_id;
	std::chrono::high_resolution_clock::time_point created, enqueued, 
		eligible_for_dequeue, dequeued, exec_complete, async_complete;
	float async_wait, async_duration;
};

struct ExecutorTelemetry {
	int task_type, executor_id;
	std::chrono::high_resolution_clock::time_point next_task_begin, slot_available, task_dequeued, task_complete;
	float async_wait, async_duration;
};

struct RequestTelemetry {
	int model_id, request_id;
	std::chrono::high_resolution_clock::time_point arrived, submitted, complete;
	std::vector<TaskTelemetry*> tasks;
};

struct SerializedTaskTelemetry {
	int task_type, executor_id, model_id, action_id;
	uint64_t created, enqueued, eligible_for_dequeue, dequeued, exec_complete, async_complete;
	uint64_t async_wait, async_duration;

	PODS_SERIALIZABLE(1,
		PODS_MDR(task_type),
		PODS_MDR(executor_id),
		PODS_MDR(model_id),
		PODS_MDR(action_id),
		PODS_MDR(created),
		PODS_MDR(enqueued),
		PODS_MDR(eligible_for_dequeue),
		PODS_MDR(dequeued),
		PODS_MDR(exec_complete),
		PODS_MDR(async_complete),
		PODS_MDR(async_wait),
		PODS_MDR(async_duration)
    )
};

struct SerializedExecutorTelemetry {
	int task_type, executor_id;
	uint64_t next_task_begin, slot_available, task_dequeued, task_complete;
	uint64_t async_wait, async_duration;

	PODS_SERIALIZABLE(1,
		PODS_MDR(task_type),
		PODS_MDR(executor_id),
		PODS_MDR(next_task_begin),
		PODS_MDR(slot_available),
		PODS_MDR(task_dequeued),
		PODS_MDR(task_complete),
		PODS_MDR(async_wait)
    )
};

struct SerializedRequestTelemetry {
	int model_id, request_id;
	uint64_t arrived, submitted, complete;
	std::vector<SerializedTaskTelemetry> tasks;

	PODS_SERIALIZABLE(1,
		PODS_MDR(model_id),
		PODS_MDR(request_id),
		PODS_MDR(arrived),
		PODS_MDR(submitted),
		PODS_MDR(complete),
		PODS_MDR(tasks)
    )
};

}

#endif