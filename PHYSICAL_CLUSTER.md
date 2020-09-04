# Running Gavel on a physical cluster

## Overview

Gavel is comprised of a centralized scheduler (deployed on a separate scheduling
server) and workers (each worker has 1 or more GPUs). Jobs are submitted
to the scheduler. The scheduler then computes a heterogeneity-aware allocation for
each active job using its policy framework. It then uses its round-based scheduling
mechanism to determine how to grant resources to jobs.

## Environment setup

To setup the environment necessary to run Gavel, simply run
`pip install -r scheduler/requirements.txt` followed by `make` -- this needs to be done
on the scheduler server as well as on all the worker servers.

## Scripts

We provide scripts to launch both the scheduler and workers.
To launch the scheduler, use `scripts/drivers/run_scheduler.py` as follows:
```bash
python scripts/drivers/run_scheduler_with_trace.py \
  --trace traces/physical_cluster/artifact_evaluation.trace \
  --seed 0 \
  --solver ECOS \
  --throughputs_file physical_cluster_throughputs.json \
  --time_per_iteration 360 \
  --policy max_min_fairness_perf \
  --expected_num_workers 4
```
Running this command will start the scheduler and log the IP address
the server is running with. Using this IP address, we can launch a worker
as follows:
```bash
python worker.py \
  -t v100 \
  -i [IP_ADDRESS] -w 50061 -g 4 \
  --run_dir /path/to/workloads \
  --data_dir /path/to/data \
  --checkpoint_dir /path/to/checkpoints
```
This should be done for all workers in the cluster.

The included trace for artifact evaluation should complete in XX hours using
a cluster size of XX.
