# Clockwork

A multi-tenant managed inference server, backed by a modified version of TVM.

This README file describes the pre-requisites and steps required to build and run Clockwork.  If you follow these steps but encounter errors, please e-mail the mailing list.

Clockwork is not feature complete, but we welcome contributions from others!

Mailing List: clockwork-users@googlegroups.com

# Resources

* [Installation Pre-Requisites](docs/prerequisites.md)
* [Building Clockwork](docs/building.md)
* [Environment Configuration](docs/environment.md)
* [Clockwork Configuration](docs/configuration.md)
* [Troubleshooting Guide](docs/troubleshooting.md)




# Step 5: Check Environment

## Check the environment is OK

From Clockwork's `build` directory,

```
./profile [check]
```

This will check your current environment settings.

## Check models run

```
./check_model /home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model
```

Path should be the path to a model, e.g. from `clockwork-modelzoo-volta` in the above example.

## Testing without using GPUs

At least 3 machines are required in order to run Clockwork (1 worker, 1 controller, 1 client).

Clockwork can run without GPUs using an emulated worker.

The binaries used below exist in Clockwork's `build` directory

### 1. Start one or more workers

```
./worker_dummy -n 2
```

Note: the `-n 2` specifies it should simulate 2 GPUs.  You can simulate many GPUs by increasing this number.

By default the worker will listen on port 12345.  Run `./worker_dummy -h` for more options.

### 2. Start the controller

```
./controller INFER4 volta01:12345
```

Here, `volta01:12345` is the address of the worker started in step 1.  `INFER4` is the name of the default Clockwork scheduler.

By default, the controller will listen for client connections on port 12346.  Run `./controller -h` for more options.

### 3. Start a client

```
./client volta02:12346 simple
```

Here, `volta02:12346` is the address of the controller started in step 2.  `simple` is the name of a workload.  Run `./client -h` to list available workloads.

### Summary

After running the above, you will see the following outputs:

Worker:
```
jcmace@volta01:~/clockwork/build$ ./worker_dummy -n 2
Starting Clockwork Worker
Loading Clockwork worker default config from /home/jcmace/clockwork/config/default.cfg
IO service thread listening on 0.0.0.0:12345
Received A0:GetWorkerState
Sending R0:GetWorkerState:
 page_size=16777216
gpus=
 GPU-0 weights_cache=21.5GB (1376 pages) io_pool=512.0MB workspace_pool=512.0MB 0 models currently on GPU
 GPU-1 weights_cache=21.5GB (1376 pages) io_pool=512.0MB workspace_pool=512.0MB 0 models currently on GPU
models=

Clock Skew=0  RTT=0  LdWts=0  Inf=0  Evct=0  || Total Pending=0  Errors=0
Received A0:LoadModelFromDisk model=0 [0.0, 16847613635838.0] /home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model
Sending R0:LoadModelFromDisk input=602112 output=4000 weights=112.0 MB (7 pages) xfer=8.4 b1=3.3 b2=5.3 b4=7.5 b8=12.4 duration=14.9
Clock Skew=673306  RTT=53853  LdWts=4  Inf=2885  Evct=0  || Total Pending=2  Errors=0
Clock Skew=668080  RTT=55526  LdWts=0  Inf=4020  Evct=0  || Total Pending=2  Errors=0
Clock Skew=665855  RTT=54409  LdWts=0  Inf=3986  Evct=0  || Total Pending=1  Errors=0
```

Controller:
```
jcmace@volta02:~/clockwork/build$ ./controller INFER4 volta01:12345
Starting Clockwork Controller
Logging requests to /local/clockwork_request_log.tsv
Logging actions to /local/clockwork_action_log.tsv
ConcurrentInferAndLoadScheduler using:
         default_slo=100000000
         latest_delta=10000000
         schedule_ahead=10000000
         max_allowable_exec_time=250000000
         max_batch_size=8
         generate_inputs=0
         max_gpus=100
IO service thread listening for clients on Connecting to worker volta01:12345
0.0.0.0:12346
Connection established
(Startup) Running ControllerStartup
(Startup-1) Bouncing LS and Infer requests until startup is complete
(Startup-2) Querying current worker state
Clockwork page_size=16777216
Worker 0 2 GPUs 0 models
GPU 0 21.5 GB (1376 pages) 0 loaded models
GPU 1 21.5 GB (1376 pages) 0 loaded models

(Startup-3) Awaiting LoadModel requests from clients
Client  --> Req0:LoadModel path=/home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model
(Startup-4) LoadModelStage has begun
Worker <--  A0:LoadModelFromDisk model=0 [0.0, 16847613635839.0] /home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model
Worker  --> R0:LoadModelFromDisk input=602112 output=4000 weights=112.0 MB (7 pages) xfer=8.4 b1=3.3 b2=5.3 b4=7.5 b8=12.4 duration=14.9
Client <--  Rsp0:LoadModel model_id=[0->1] input=602112 output=4000
Client  --> Req1:LS
(Startup-6) LoadModelStage complete.  Printing loaded models:
Clockwork page_size=16777216
Worker 0 2 GPUs 2 models
GPU 0 21.5 GB (1376 pages) 0 loaded models
GPU 1 21.5 GB (1376 pages) 0 loaded models
M-0 src=/home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model input=602112 output=4000 weights=112.0 MB (7 pages) xfer=8.4 b1=3.3 b2=5.3 b4=7.5 b8=12.4
M-1 src=/home/jcmace/clockwork-modelzoo-volta/resnet50_v2/model input=602112 output=4000 weights=112.0 MB (7 pages) xfer=8.4 b1=3.3 b2=5.3 b4=7.5 b8=12.4

(Startup-end) Transitioning to scheduler
Client <--  Rsp1:LS error 5: Controller initializing
Created 2 models
Created 2 GPUs on 1 Workers
Total GPU capacity 2752 pages (1376 per GPU).
Total model pages 14 (0% oversubscription).
 * Admitting inference requests
GPU handler [ 0 ] started
GPU handler [ 1 ] started
Network Status:  Client ✔✔✔✔✔ Controller ✔✔✔✔✔ Workers (inputs generated by client)
W0-GPU0 LoadW min=8.37 max=8.37 mean=8.37 e2emean=9.67 e2emax=9.67 throughput=0.0 utilization=0.00 clock=[0-0] norm_max=0.00 norm_mean=0.00
Client throughput=0.0 success=100.00% min=13.9 max=13.9 mean=13.9
Network->Workers: 31.4MB/s (361 msgs) snd, 1.4MB/s (361 msgs) rcv,
W0-GPU0 LoadW min=8.37 max=8.37 mean=8.37 e2emean=17.80 e2emax=17.80 throughput=0.1 utilization=0.00 clock=[0-0] norm_max=0.00 norm_mean=0.00
W0-GPU0 Infer min=3.32 max=3.32 mean=3.32 e2emean=4.47 e2emax=7.25 throughput=200.6 utilization=0.67 clock=[1380-1380] norm_max=3.32 norm_mean=3.32
Client throughput=400.8 success=100.00% min=3.6 max=21.8 mean=4.7
Network->Workers: 34.8MB/s (402 msgs) snd, 1.6MB/s (402 msgs) rcv,
W0-GPU1 LoadW min=8.37 max=8.37 mean=8.37 e2emean=13.73 e2emax=17.79 throughput=0.2 utilization=0.00 clock=[0-0] norm_max=0.00 norm_mean=0.00
W0-GPU1 Infer min=3.32 max=3.32 mean=3.32 e2emean=4.47 e2emax=7.13 throughput=200.1 utilization=0.66 clock=[1380-1380] norm_max=3.32 norm_mean=3.32
```

Client:

```
jcmace@volta03:~/clockwork/build$ ./client volta02:12346 simple
Running workload `simple` on volta02:12346
Client is sending inputs with requests.  Set CLOCKWORK_DISABLE_INPUTS=1 to disable inputs.
Connecting to clockwork @ volta02:12346
Connection established
Found 61 models in /home/jcmace/clockwork-modelzoo-volta
Clockwork initializing, retrying Controller initializing
throughput=300.20 min=3.82 max=22.49 mean=4.97
throughput=401.83 min=3.82 max=7.74 mean=4.96
throughput=398.53 min=3.82 max=7.86 mean=5.00
```

## Testing with GPUs

Repeat the above steps, but instead of running `worker_dummy`, run `worker`.  To see the options available, run `worker -h`.  By default, `worker` will use all available GPUs.

The outputs from each process should be the same.



