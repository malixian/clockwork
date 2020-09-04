# Clockwork

A multi-tenant managed inference server, backed by a modified version of TVM.

This README file describes the pre-requisites and steps required to build and run Clockwork.  If you follow these steps but encounter errors, please e-mail the mailing list.

Clockwork is not feature complete, but we welcome contributions from others!

Mailing List: clockwork-users@googlegroups.com

# Resources

## Other Repositories

The following other repositories are relevant and will be referenced here and there.

* [`clockwork-results`](https://gitlab.mpi-sws.org/cld/ml/clockwork-results) contains experiment scripts and documentation for reproducing results from the OSDI 2020 Clockwork paper.
* [`clockwork-modelzoo-volta`](https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta) contains pre-compiled models that can be used for experimentation
* [`azure-functions`](https://gitlab.mpi-sws.org/cld/trace-datasets/azure-functions) contains workload traces from Microsoft Azure that can be used for experimentation

## Getting Started Guide

* [Installation Pre-Requisites](docs/prerequisites.md)
* [Building Clockwork](docs/building.md)
* [Environment Setup](docs/environment.md)
* [Clockwork Configuration](docs/configuration.md)
* [Running Clockwork for the first time](docs/firstrun.md)

## Other Resources
* [Troubleshooting Guide](docs/troubleshooting.md)



## Testing with GPUs

Repeat the above steps, but instead of running `worker_dummy`, run `worker`.  To see the options available, run `worker -h`.  By default, `worker` will use all available GPUs.

The outputs from each process should be the same.



