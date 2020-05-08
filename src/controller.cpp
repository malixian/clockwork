// #include "clockwork/network/controller.h"
// #include "clockwork/controller/direct_controller.h"
// #include "clockwork/controller/closed_loop_controller.h"
// #include "clockwork/controller/stress_test_controller.h"

#include "gurobi_c++.h"
#include <cstdlib>
#include <algorithm>    // std::random_shuffle
#include <sstream>

// using namespace clockwork;

// int main(int argc, char *argv[]) {
// 	std::cout << "Starting Clockwork Controller" << std::endl;

// 	if ( argc < 4) {
// 		std::cerr << "USAGE ./controller [CLOSED_LOOP/DIRECT/STRESS/ECHO] MAX_BATCH_SIZE worker1:port1 worker2:port2 ..." << std::endl;
// 		return 1;
// 	}
	
// 	std::string controller_type = argv[1];

// 	int batch_size = atoi(argv[2]);

// 	int client_requests_listen_port = 12346;


// 	std::vector<std::pair<std::string, std::string>> worker_host_port_pairs;
// 	for (int i = 3; i < argc; i++) {
// 		std::string addr = std::string(argv[i]);
// 		auto split = addr.find(":");
// 		std::string hostname = addr.substr(0, split);
// 		std::string port = addr.substr(split+1, addr.size());
// 		worker_host_port_pairs.push_back({hostname, port});
// 	}

// 	if ( controller_type == "CLOSED_LOOP"){
// 		ClosedLoopControllerImpl* controller = new ClosedLoopControllerImpl(client_requests_listen_port, worker_host_port_pairs, batch_size);
// 		controller->join();
// 	} else if (controller_type == "DIRECT") {
// 		DirectControllerImpl* controller = new DirectControllerImpl(client_requests_listen_port, worker_host_port_pairs);
// 		controller->join();
// 	} else if (controller_type == "STRESS") {
// 		StressTestController* controller = new StressTestController(client_requests_listen_port, worker_host_port_pairs);
// 		controller->join();
// 	} else if (controller_type == "ECHO") {
// 		Scheduler* scheduler = new EchoScheduler();
// 		controller::ControllerWithStartupPhase* controller = new controller::ControllerWithStartupPhase(
// 			client_requests_listen_port,
// 			worker_host_port_pairs,
// 			10000000000UL, // 10s load stage timeout
// 			10, // 10 profiling iterations
// 			new controller::ControllerStartup(), // in future the startup type might be customizable
// 			scheduler
// 		);
// 		controller->join();
// 	}

// 	std::cout << "Clockwork Worker Exiting" << std::endl;
// }
/* Copyright 2020, Gurobi Optimization, LLC */

/* This example formulates and solves the following simple MIP model:

     maximize    x +   y + 2 z
     subject to  x + 2 y + 3 z <= 4
                 x +   y       >= 1
                 x, y, z binary
*/

#include "gurobi_c++.h"
using namespace std;

class Request;
class Worker;

class RequestOnWorker {
public:
  std::string name;
  Request* req;
  Worker* worker;
  GRBVar var;
  bool ran = false;
  RequestOnWorker(Request* req, Worker* worker, std::string name) : req(req), worker(worker), name(name) {
  }
  void addToModel(GRBModel &model, float size) {
    var = model.addVar(0.0, 1.0, 1.0, GRB_BINARY);
    if (ran) var.set(GRB_DoubleAttr_Start, 1.0);
  }
  void printIfAssigned(float size) {
    if (var.get(GRB_DoubleAttr_X) == 1.0) {
      std::cout << name << " " << size << std::endl;
    }
  }
};

class Request {
public:
  unsigned id;
  float size;
  float deadline;
  bool ran = false;
  std::vector<RequestOnWorker*> instances;
  Request(unsigned request_id, float size, float deadline) : id(request_id), size(size), deadline(deadline) {}
  void addToModel(GRBModel &model) {
    GRBLinExpr expr = 0;
    for (auto &r : instances) {
      expr += r->var;
    }
    model.addConstr(expr <= 1);
  }
};

class Worker {
public:
  unsigned id;
  float available;
  float sched_available = 0;
  std::vector<RequestOnWorker*> queue;
  Worker(unsigned worker_id) : id(worker_id), available(0) {}
  void addToModel(GRBModel &model) {
    for (unsigned i = 0; i < queue.size(); i++) {
      GRBLinExpr expr = available;
      for (unsigned j = 0; j <= i; j++) {
        expr += (queue[j]->var * queue[j]->req->size);
      }
      std::stringstream s;
      s << "Execute Request " << queue[i]->req->id << " on Worker " << id;
      model.addConstr(expr <= (queue[i]->req->deadline + queue[i]->req->size), s.str());
    }
  }
};

class Clockwork {
public:
  std::vector<Request*> requests;
  std::vector<Worker*> workers;
  std::vector<RequestOnWorker*> instances;

  void addWorker() {
    workers.push_back(new Worker(workers.size()));
  }

  void enqueue(float size, float deadline, std::vector<unsigned> workerids) {
    std::stringstream ss;
    ss.precision(1);
    ss << std::fixed;
    ss << "Add r-" << requests.size() << " size=" << size << " deadline=" << deadline << " on " << workerids[0];
    for (unsigned i = 1; i < workerids.size(); i++) {
      ss << "," << workerids[i];
    }
    std::cout << ss.str() << std::endl;

    float min_worker_t = deadline-size;
    Worker* min_worker = nullptr;
    RequestOnWorker* min_rw = nullptr;


    Request* r = new Request(requests.size(), size, deadline);
    for (unsigned worker_id : workerids) {
      Worker* w = workers[worker_id];

      std::stringstream s;
      s << "r-" << r->id << "-w-" << w->id;
      std::string name = s.str();

      RequestOnWorker* rw = new RequestOnWorker(r, w, name);

      r->instances.push_back(rw);
      w->queue.push_back(rw);
      instances.push_back(rw);

      if (w->sched_available <= min_worker_t) {
        min_worker_t = w->sched_available;
        min_worker = w;
        min_rw = rw;
      }
    }
    requests.push_back(r);

    if (min_worker != nullptr) {
      min_rw->ran = true;
      r->ran = true;
      min_worker->sched_available += size;
    }
  }

  void addToModel(GRBModel &model) {
    GRBLinExpr obj = 0;
    for (RequestOnWorker* rw : instances) {
      rw->addToModel(model, rw->req->size);
      obj += rw->var;
    }
    for (Request* r : requests) {
      r->addToModel(model);
    }
    for (Worker* w : workers) {
      w->addToModel(model);
    }
    // model.setObjective(obj, GRB_MAXIMIZE);
  }

  void printAssignments() {
    for (auto rw : instances) {
      rw->printIfAssigned(rw->req->size);
    }
  }

  void printFifoObjective() {
    float total = 0;
    for (Request* r : requests) {
      if (r->ran) total += r->size;
    }
    std::cout << "Fifo objective = " << total << std::endl;
  }
};

std::vector<unsigned> assign(unsigned num_workers) {
  std::vector<bool> loaded(num_workers, false);

  loaded[0] = true;
  for (unsigned i = 1; i < num_workers; i++) {
    if (rand() % 2 == 0) break;
    loaded[i] = true;
  }

  std::random_shuffle(loaded.begin(), loaded.end());
  std::vector<unsigned> result;
  for (unsigned i = 0; i < loaded.size(); i++) {
    if (loaded[i]) result.push_back(i);
  }
  return result;
}

float myrand() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

int
main(int   argc,
     char *argv[])
{
  try {
    srand(0);

    float deadline_min = 20;
    float deadline_max = 100;
    float size_min = 1;
    float size_max = 9;
    int num_workers = 24;
    int num_requests = 100;

    float load = (num_requests * ((size_max - size_min) / 2.0)) / ((deadline_max - deadline_min) * num_workers);

    std::cout << "Adding " << num_requests << " requests on " << num_workers << " workers" << std::endl;
    std::cout << "Estimated load " << load << std::endl;

    Clockwork clockwork;
    for (unsigned i = 0; i < num_workers; i++) {
      clockwork.addWorker();
    }

    for (unsigned i = 0; i < num_workers; i++) {
      clockwork.enqueue(80, 81, {i});
    }
    for (unsigned i = 0; i < num_requests; i++) {
      std::vector<unsigned> workerids = assign(clockwork.workers.size());
      float deadline = deadline_min + i * (deadline_max - deadline_min) / num_requests;
      float size = size_min + myrand() * (size_max - size_min);
      clockwork.enqueue(size, deadline, workerids);
    }


    // Create an environment
    GRBEnv env = GRBEnv(true);
    // env.set("LogFile", "mip1.log");
    env.start();

    // Create an empty model
    GRBModel model = GRBModel(env);

    clockwork.addToModel(model);

    // model.set(GRB_IntParam_Presolve, 0);
    // model.set(GRB_DoubleParam_Heuristics, 0.5);
    // model.set(GRB_IntParam_StartNodeLimit, 2000000000);
    model.set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);
    model.set(GRB_DoubleParam_TimeLimit, 0.1);
    // model.set(GRB_IntParam_MIPFocus, 3);
    // model.set(GRB_IntParam_SubMIPNodes, 10000);
    // model.set(GRB_DoubleAttr_Start, GRB_UNDEFINED);

    model.optimize();
      int status = model.get(GRB_IntAttr_Status);

  if ((status == GRB_INF_OR_UNBD) ||
      (status == GRB_INFEASIBLE)  ||
      (status == GRB_UNBOUNDED)     ) {
    cout << "The model cannot be solved " <<
    "because it is infeasible or unbounded" << endl;
  }
  if (status != GRB_OPTIMAL) {
    cout << "Optimization was stopped with status " << status << endl;
  }

    // // Create variables
    // GRBVar x = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x");
    // GRBVar y = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "y");
    // GRBVar z = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "z");

    // // Set objective: maximize x + y + 2 z
    // model.setObjective(x + y + 2 * z, GRB_MAXIMIZE);

    // // Add constraint: x + 2 y + 3 z <= 4
    // model.addConstr(x + 2 * y + 3 * z <= 4, "c0");

    // // Add constraint: x + y >= 1
    // model.addConstr(x + y >= 1, "c1");

    // // Optimize model
    // model.optimize();

    // cout << x.get(GRB_StringAttr_VarName) << " "
    //      << x.get(GRB_DoubleAttr_X) << endl;
    // cout << y.get(GRB_StringAttr_VarName) << " "
    //      << y.get(GRB_DoubleAttr_X) << endl;
    // cout << z.get(GRB_StringAttr_VarName) << " "
    //      << z.get(GRB_DoubleAttr_X) << endl;

    // cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

    // clockwork.printAssignments();
    clockwork.printFifoObjective();

    cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }

  return 0;
}