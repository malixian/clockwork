# Customizing Your Environment

Clockwork was developed on machines with the following configuration:

* 2x10Gbit network links (bonded to 20Gbit)
* 32 cores
* 768GB RAM
* 2x 32GB nVidia Tesla v100 GPUs per Worker

This page describes what you can expect and what you will need to do if you run Clockwork in a **different** environment.

## Cores

Clockwork's worker requires `3+9*{gpu_count}` CPU cores.  So for a node with 1 GPU, a worker requires 12 cores.

Clockwork's controller requires approximately 18 cores.

The above are hard limits.

## RAM

Clockwork's worker requires increasing RAM depending on the number of models loaded.

For example, using approx. 4000 models required around 500GB RAM.

Depending on the workload you wish to run, you will need to ensure you have enough RAM.  The amount of RAM needed by a model is approximately equal to the size of its weights.

Alternatively, for most workloads, you can scale down the number of models used, to fit the RAM you have available.
*  Run `./client -h` to read workload descriptions; most workloads are parameterized and can be adjusted to use fewer models
*  The experiments in the [`clockwork-results`](https://gitlab.mpi-sws.org/cld/ml/clockwork-results) describe how you can adjust the workloads to use fewer models and less RAM.

## GPU Architecture

Clockwork can work on any GPU architecture.  However, the models in [`clockwork-modelzoo-volta`](https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta) are specifically compiled for `sm_70` (Tesla) architectures.

Providing instructions for compiling models for other architectures is TODO.

## GPU Memory

Clockwork was developed and run on machines with 32GB Tesla v100 GPUs.  However, many cloud providers commonly offer 16GB Tesla v100 GPUs.

Clockwork can run with lower GPU memory; however, it requires configuration changes:

* Edit `config/default.cfg` on worker machines and set `weights_cache_size` to 10737418240L.
* The maximum number of models will be approximately 2000.  Some of the experiments in [`clockwork-results`](https://gitlab.mpi-sws.org/cld/ml/clockwork-results) will not be able to run with a full complement of models.


#### Troubleshooting:
* A `CUDA: out of memory` from `src/clockwork/cache.cpp` indicates that `weights_cache_size` is too large for your GPU.
* A `CUDA_ERROR_OUT_OF_MEMORY` from `src/clockwork/model/cuda.cpp` indicates too many models loaded.  You can either reduce the number of models you use, or reduce the value of `weights_cache_size` (which by proxy increases memory available for kernels).

## Network

Clockwork experiments were performed in a private cluster.  We have not extensively experimented with Clockwork in a cloud setting, and issues may arise due to network contention that we are unfamiliar with.

A network less than 10Gbit is not recommended and Clockwork may not work properly.  Clockwork's controller only expects a small amount of latency for each action on the network.  If your network is too slow, all actions will fail.  If you see `throughput=0`, and if your worker nodes are only reporting errors, then network latency is an issue.  (Note: this is solvable by removing some hard-coding in the controller.  Please consider contributing!)


