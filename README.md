# Clockwork

A multi-tenant managed inference server, backed by a modified version of TVM.

This README file describes the pre-requisites and steps required to build and run Clockwork.  If you follow these steps but encounter errors, please e-mail the mailing list.

Clockwork is not feature complete, but we welcome contributions from others!

Mailing List: clockwork-users@googlegroups.com

# Step 1: Pre-Requisites


## 1. NVIDIA Driver and CUDA 


Make sure NVIDIA driver and CUDA are installed and CUDA is on your PATH. MPI cluster machines have CUDA 9 installed by default. You can check if CUDA is installed and the version by running `nvcc --version`

## 2. Required Packages

The following apt packages pre-requisites:

```
apt install libtbb-dev libasio-dev libconfig++-dev g++-8 \
make cmake automake autoconf libtool curl unzip clang llvm
```
## 3. Installing Protobuf

```
git clone --recursive -b v3.12.0 https://github.com/protocolbuffers/protobuf.git
cd protobuf
./autogen.sh && ./configure 
make -j $(nproc) 
make install
ldconfig
cd ..
```

## 4. Installing TVM

Clone our modified TVM and check out our modified branch (`clockwork-v0.6`):
```
git clone --recursive -b clockwork-v0.6 https://gitlab.mpi-sws.org/cld/ml/tvm
```

Build TVM
```
cd tvm/build
cmake ..
make -j $(nproc)
cd ..
```

Set `TVM_HOME` environment variable and add `$TVM_HOME/build` to your `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` environment variables
```
echo "export TVM_HOME=`pwd`" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TVM_HOME/build" >> ~/.bashrc
echo "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$TVM_HOME/build" >> ~/.bashrc
source ~/.bashrc
```

# Step 2: Building Clockwork

```
git clone --recursive https://gitlab.mpi-sws.org/cld/ml/clockwork.git
cd clockwork
mkdir build
cd build
cmake ..
make -j $(nproc)
```


# Step 3: Environment Configuration

Clockwork is a high-performance system that depends upon predictability.  There are various tweaks to your environment that will make executions more predictable.  These environment modifications should be made for Clockwork's worker, controller, and client processes. Some are optional but recommended.

## Check your environment

You can check your environment by running Clockwork's:
```
./profile [check]
```

## 1. Increase resource limits (memlock, nofile, rtprio)

Limits on the number of open files, and the amount of page-locked memory, reduce the total number of DNNs clockwork can keep in memory at any point in time.  A limit of 1024 is too low.  A limit of 16k or higher is acceptable.

Limits can be checked with the `ulimit` command (`ulimit -aH` lists hard limits, `ulimit -a` lists current)

Increase the `RLIMIT_NOFILE` (number of open files) and `RLIMIT_MEMLOCK` (amount of page-locked memory) to unlimited:
1. Open `/etc/security/limits.conf`
2. Add the following lines:
```
*            hard   memlock           unlimited
*            soft   memlock           unlimited
*            hard   nofile            unlimited
*            soft   nofile            unlimited
*            hard   rtprio            unlimited
*            soft   rtprio            unlimited
```
Note: for MPI cluster machines with the default Debian distribution, you will also need to modify `/etc/security/limits.d/mpiprefs.conf`

3. Modify `/etc/systemd/user.conf` and `/etc/systemd/system.conf` to add:
```
DefaultLimitNOFILE=1048576
```
4. Restart to take effect
5. Upon restarting, use Clockwork's `./profile [check]` to check if the settings took effect

## 2. Increase mmap limits

Clockwork uses a lot of shared objects, and we need to increase the mmap limit.  As root, run
```
/usr/sbin/sysctl -w vm.max_map_count=10000000
```

In general you can check mmap limits with:
```
sysctl vm.max_map_count
```

This normally does not require a restart.  You can check using Clockwork's `./profile [check]`.

This normally does not require a restart.  You can check using Clockwork's `./profile [check]`.

## 3. GPU Settings

## 3.1. Disable CUDA JIT

Prevent CUDA from caching compiled kernels (note: the models used by Clockwork do not compile to PTX anyway, but if choose to compile JITable models, this setting is important)
```
export CUDA_CACHE_DISABLE=1
```

### 3.1 Enable persistence mode.

```
nvidia-smi -pm 1
```

NOTE: This must be done on every restart

### 3.2 Enable exclusive process mode

```
nvidia-smi -c 3
```

NOTE: This must be done on every restart


### 3.3 Optional: Disable auto boost

```
nvidia-smi --auto-boost-default=DISABLED
```

NOTE: This must be done on every restart

### 3.4 Optional: Configure GPU clocks

You can specify which clock frequencies to use.  This does not override built-in temperature auto-scaling.

List available GPU clock frequencies
```
nvidia-smi -q -d SUPPORTED_CLOCKS
```

Pick a memory and graphics clock frequency (usually the highest), e.g. on volta machines:
```
    Supported Clocks
        Memory                      : 877 MHz
            Graphics                : 1380 MHz
```

Set the default application clock and system clock to those highest values, e.g. on volta machines:
```
nvidia-smi -ac 877,1380
nvidia-smi -lgc 1380
```

NOTE: This must be done on every restart

## 4. Optional: Disable CPU frequency autoscaling

Set the "performance" governor to prevent CPU clock scaling
```
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

# Step 4: Runtime Environment Variables

In addition to the environment setup above, Clockwork has several environment variables of its own.

## Required: CLOCKWORK_MODEL_DIR

This is required by Clockwork's `./worker` process.

`CLOCKWORK_MODEL_DIR` should point to a local directory containing compiled models.

Pre-compiled models used by Clockwork can be found at the `clockwork-modelzoo-volta` repository: https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta

The process of compiling models is not fully automated currently (please contribute!).

## Recommended: CLOCKWORK_LOG_DIR

This is required by Clockwork's `./controller` process.

`CLOCKWORK_LOG_DIR` should point to a directory where the controller can write its output request logs.  Be aware that for long experiments, these files can be GB large.

If not specified or left blank, Clockwork's controller will write to `/local`.  If this does not exist on your machine or is not writable, the controller will not output anything.

Please ensure the directory exists; Clockwork will not create the directory for you.  Upon startup, the controller will print the location it is writing its request and action logs to.

## Optional: AZURE_TRACE_DIR

This is required by Clockwork's `./client` process if you are running the `azure` workload.

`AZURE_TRACE_DIR` should point to a local directory containing the `AzureFunctionsDataset2019` from Microsoft Azure.

The original traces can be found by following the instructions on Microsoft's GitHub repository: https://github.com/Azure/AzurePublicDataset.

Alternatively, a repository containing the traces can be found here: https://gitlab.mpi-sws.org/cld/trace-datasets/azure-functions.

## Optional: CLOCKWORK_DISABLE_INPUTS

This is used by Clockwork's `./client` process.

For some experiments you will want to generate model inputs at the controller rather than sending them over the network.

Setting `CLOCKWORK_DISABLE_INPUTS=1` will disable clients from sending inputs.

## Optional: CLOCKWORK_CONFIG_FILE

This is used by Clockwork's `./worker` process.

Clockwork has a configuration file located under `config/default.cfg`.  You can specify your own configuration elsewhere and set its path using `CLOCKWORK_CONFIG_FILE`.

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

# Troubleshooting

## Cannot find nvidia-ml

Currently, the CMakeLists assumes CUDA lives in either `/usr/local/cuda/lib64` (the default location in Ubuntu 14.x) or `/usr/lib/x86_64-linux-gnu/nvidia/current` (the default location for MPI cluster machines).  If you get build errors saying cannot find CUDA or cannot find nvidia-ml, then you'll need to update the `include_directories` and `link_directories` directives in the CMakeLists.txt with the CUDA location on your machine.

## Undefinied reference to tvm::runtime::ManagedCuda...

Undefined reference to tvm::runtime::ManagedCuda... -- this probably means you didn't build TVM properly.  Make sure you haven't modified or deleted the file `build/config.cmake` in the TVM repository.  `make clean` and `make` TVM again.

## Unable to set number of open files with ulimit

Unable to set number of open files with ulimit: default values are picked up from conf files, e.g. /etc/security/limits.conf, but they may be overwritten by files in a subdirectory, e.g. /etc/security/limits.d/mpi.conf

Make sure, upon restarting, that the correct ulimit values have been set, by running `./profile [check]`

## Cannot apply memory protection

If you are loading lots of models, you might see the following:
*  `src/clockwork/model/so.cpp:20: Check failed: lib_handle_ != nullptr: Failed to load SO /proc/26344/fd/14656/proc/26344/fd/14656: cannot apply additional memory protection after relocation: Cannot allocate memory`
*  `src/clockwork/model/so.cpp:20: Check failed: lib_handle_ != nullptr: Failed to load SO /proc/12386/fd/11804/proc/12386/fd/11804: failed to map segment from shared object`

Make sure your `mmap` limits have been correctly set as described above.  You can check by running `./profile [check]`

## CUDA: out of memory

Upon starting a worker process, you might see:
```
  what():  [20:58:09] /home/jcmace/clockwork/src/clockwork/cache.cpp:237: Check failed: e == cudaSuccess || e == cudaErrorCudartUnloading: CUDA: out of memory
```

If this happens, make sure the `weights_cache_size` is set to an appropriate value in `config/default.cfg`.  By default, these values are configured assuming a 32GB v100 GPU.  If you are using a 16GB v100 GPU, you need to reduce these values.  For a 16GB GPU, `10737418240L` is an appropriate value for `weights_cache_size`.

## cuModuleLoadData Error: CUDA_ERROR_OUT_OF_MEMORY

Clockwork cannot load infinitely many models.  Each model requires up to 1MB for its kernels on the GPU.  With thousands of models this can add up!

While loading models, the client may exit with:
```
  what():  [21:03:51] /home/jcmace/clockwork/src/clockwork/model/cuda.cpp:80: cuModuleLoadData Error: CUDA_ERROR_OUT_OF_MEMORY
```

If this happens, the GPU ran out of memory.  To fix it, you can:
* Use fewer models for your experiment
* Reduce the amount of GPU memory used for `weights_cache_size`.  You can modify this in `config/default.cfg` on workers.  Reducing weights cache will leave more memory available for kernels.
