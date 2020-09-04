# Running Gavel on a physical cluster

## Overview

Gavel is comprised of a `Scheduler` deployed on the main scheduling
server and one `Worker` deployed on each GPU server. Jobs are submitted
to the `Scheduler` which computes a heterogeneity-aware scheduling policy
and then deploys jobs accordingly to the `Worker`(s) in rounds.

## Environment setup

To setup the environment necessary to run Gavel, simply run
`pip install -r requirements.txt` followed by `make` on the
scheduler server as well as all worker servers in the cluster.

## Scripts

Gavel provides scripts to launch both the `Scheduler` and the `Worker`s.
To launch a `Scheduler`, use `scripts/drivers/run_scheduler.py` as follows:
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
Running this command will start the `Scheduler` and log the IP address
the server is running with. Using this IP address, we can launch a `Worker`
as follows:
```bash
python worker.py \
  -t v100 \
  -i [IP_ADDRESS] -w 50061 -g 4 \
  --run_dir /path/to/workloads \
  --data_dir /path/to/data
  --checkpoint_dir /path/to/checkpoints
```

The included trace for artifact evaluation should complete in XX hours using
a cluster size of XX. Note that while jobs will write out checkpoints
at the end of every micro-task, these checkpoints will not be synchronized
between workers as this requires configuring an NFS deployment which we do
not cover here.
