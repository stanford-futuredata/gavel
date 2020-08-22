# Gavel Scheduler

This directory contains implementation for the Gavel scheduler, including
its heterogeneous-aware policy framework, its scheduling mechanism, and a runtime
and simulator.

- `policies/`: Implementations of various heterogeneous-aware policies as well
  as baselines.
- `scheduler.py`: Scheduling mechanism and simulator for Gavel.
- `runtime/`: RPC implementation between scheduler and workers.
- `gavel_iterator.py`: Implementation of `GavelIterator` that facilitates
  seamless time sharing of deep learning training applications.
- `scripts/`: Various scripts to run experiments and deploy the scheduler.
