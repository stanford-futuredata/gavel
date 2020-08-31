# Running Gavel on a physical cluster

## Overview

Gavel is comprised of a `Scheduler` deployed on the main scheduling
server and one `Worker` deployed on each GPU server. Jobs are submitted
to the `Scheduler` which computes a heterogeneity-aware scheduling policy
and then deploys jobs accordingly to the `Worker`s in rounds.

## Environment setup

Gavel requires an NFS deployment to share models between different workers.
Instructions for how to configure NFS can be found [here](TODO:insert link).

Within the NFS root directory, Gavel requires three subdirectories:
1. A directory for the model code (e.g. `workloads`).
2. A directory for the training data (e.g. `data`).
3. A directory for model checkpoints (e.g. `checkpoints`).

## Scripts

Gavel provides scripts to launch both the `Scheduler` and the `Worker`s.
To launch a `Scheduler`, use `scripts/drivers/run_scheduler.py` as follows:
```bash
TODO: give example
```
To launch a `Worker`, use `worker.py`:
```bash
TODO: give example
```

## API

Gavel exposes the following public functions to interface with the `Scheduler`:

`add_job`: TODO

`remove_job`: TODO

TODO: flesh this out
