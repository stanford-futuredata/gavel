# Running Gavel on a physical cluster

## Overview

Gavel is comprised of a `Scheduler` deployed on the main scheduling
server and one `Worker` deployed on each GPU server. Jobs are submitted
to the `Scheduler` which computes a heterogeneity-aware scheduling policy
and then deploys jobs accordingly to the `Worker`s in rounds.

## Environment setup

Gavel requires an NFS deployment to share models between different workers.
Instructions for how to configure NFS can be found
[here](https://www.tecmint.com/install-nfs-server-on-ubuntu/).

Within the NFS root directory, Gavel requires three subdirectories:
1. A directory for the model code (e.g. `workloads`).
2. A directory for the training data (e.g. `data`).
3. A directory for model checkpoints (e.g. `checkpoints`).

## Scripts

Gavel provides scripts to launch both the `Scheduler` and the `Worker`s.
To launch a `Scheduler`, use `scripts/drivers/run_scheduler.py` as follows:
```bash
python scripts/drivers/run_scheduler_with_trace.py \
  --trace traces/physical_cluster/debug.trace \
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
