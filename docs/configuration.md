# Clockwork Configuration

## Summary

### Required Configuration

1. CLOCKWORK_MODEL_DIR (on workers)
2. CLOCKWORK_LOG_DIR (on controller)

### Optional Configuration

1. AZURE_TRACE_DIR (on client, if using `azure` workload)
2. CLOCKWORK_DISABLE_INPUTS (on client, depending on experiment)
3. CLOCKWORK_CONFIG_FILE (on workers, if overriding defaults from `config/defaults.cfg`)

## Details

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

