# Controller

Run `./controller -h` to see available controller options

```
USAGE:
  controller [TYPE] [WORKERS] [OPTIONS]
DESCRIPTION
  Run the controller of the given TYPE. Connects to the specified workers. All
  subsequent options are controller-specific and passed to that controller.
TYPE
  DIRECT    Used for testing
  ECHO      Used for testing
  STRESS    Used for testing
  INFER4    The Clockwork Scheduler.  You should usually be using this.  Options:
       generate_inputs    (bool, default false)  Should inputs and outputs be generated if not present.  Set to true to test network capacity
       max_gpus           (int, default 100)  Set to a lower number to limit the number of GPUs.
       schedule_ahead     (int, default 10000000)  How far ahead, in nanoseconds, should the scheduler schedule.  If generate_inputs is set to true, the default value for this is 15ms, otherwise 5ms.
       default_slo        (int, default 100000000)  The default SLO to use if client's don't specify slo_factor.  Default 100ms
       max_exec        (int, default 25000000)  Don't use batch sizes >1, whose exec time exceeds this number.  Default 25ms
       max_batch        (int, default 8)  Don't use batch sizes that exceed this number.  Default 8.
WORKERS
  Comma-separated list of worker host:port pairs.  e.g.:
    volta03:12345,volta04:12345,volta05:12345
OPTIONS
  -h,  --help
        Print this message
All other options are passed to the specific scheduler on init
```

