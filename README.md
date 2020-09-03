# Clockwork

A multi-tenant managed inference server, backed by a modified version of TVM.

This README file describes the pre-requisites and steps required to build and run Clockwork.  If you follow these steps but encounter errors, please e-mail the mailing list.

Clockwork is not feature complete, but we welcome contributions from others!

Mailing List: clockwork-users@googlegroups.com

# Pre-requisites

## 1. CUDA

Make sure CUDA is installed and on your PATH.  MPI cluster machines have CUDA 9 installed by default.  You can check if CUDA is installed and the version by running `nvcc --version`

## 2. Installing TVM

Clone our modified TVM and check out our modified branch:
```
git clone --recursive https://gitlab.mpi-sws.org/cld/ml/tvm
cd tvm
git checkout clockwork
```

Build TVM
```
cd build
cmake ..
make -j40
cd ..
```

Set `TVM_HOME` environment variable
```
echo "export TVM_HOME=`pwd`" >> ~/.bashrc
source ~/.bashrc
```

Add `$TVM_HOME/build` to your `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` environment variables

## 3. Apt packages

The following apt packages pre-requisites:

Intel Threading Building Blocks
```
apt-get install libtbb-dev
apt-get install libasio-dev
apt-get install libconfig++
```

# Building Clockwork

```
mkdir build
cd build
cmake ..
make -j40
```

Set the environment variable `CLOCKWORK_CONFIG_FILE` to override the default path of Clockwork worker configurations. An example configuration file is given in `config/default.cfg`

# Required Environment Modifications

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
DefaultLimitNOFILE=65535
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

# Environment Variables

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

# Troubleshooting

Currently, the CMakeLists assumes CUDA lives in either `/usr/local/cuda/lib64` (the default location in Ubuntu 14.x) or `/usr/lib/x86_64-linux-gnu/nvidia/current` (the default location for MPI cluster machines).  If you get build errors saying cannot find CUDA or cannot find nvidia-ml, then you'll need to update the `include_directories` and `link_directories` directives in the CMakeLists.txt with the CUDA location on your machine.

Undefined reference to tvm::runtime::ManagedCuda... -- this probably means you didn't build TVM properly.  Make sure you haven't modified or deleted the file `build/config.cmake` in the TVM repository.  `make clean` and `make` TVM again.

Unable to set number of open files with ulimit: default values are picked up from conf files, e.g. /etc/security/limits.conf, but they may be overwritten by files in a subdirectory, e.g. /etc/security/limits.d/mpi.conf

If you are loading lots of models, you might see the following:
*  `src/clockwork/model/so.cpp:20: Check failed: lib_handle_ != nullptr: Failed to load SO /proc/26344/fd/14656/proc/26344/fd/14656: cannot apply additional memory protection after relocation: Cannot allocate memory`
*  `src/clockwork/model/so.cpp:20: Check failed: lib_handle_ != nullptr: Failed to load SO /proc/12386/fd/11804/proc/12386/fd/11804: failed to map segment from shared object`
To solve this you need to increase your mmap limits, described above.


