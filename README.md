# Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads

This repository contains the source code implementation of the OSDI paper
"Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads".

## Directory Structure

### `scheduler`
Code for the scheduler, including the scheduling mechanism and simulator
(`scheduler.py`), implementations of performance-aware policies (`policies/`),
`GavelIterator` as a Python module, and a communication stack between the scheduler
and workers that uses [gRPC](https://grpc.io/) (`runtime/`).

### `workloads`
Implementations of target workloads in PyTorch, including changes needed to
integrate with the `GavelIterator`.


## Setup

### Software Dependencies

To run Gavel, a few Python packages are needed. These can be installed using

```bash
pip install -r scheduler/requirements.txt
```


## Reproducing Experiments

Gavel's heterogeneity-aware policies and scheduling mechanism can be evaluated
either in simulation or on a physical cluster.

To evaluate variants of the LAS policy (`max_min_fairness*`) in
simulation, one can use the following command line (this sweep script runs
the different policies for multiple traces, generated using different seeds
and Poisson arrival rates):

```bash
python -u scripts/sweeps/run_throughput_vs_latency_lambda_sweep.py -s 4000 -e 5000 -l <LOG_DIRECTORY> -j <NUM_CORES> -p max_min_fairness max_min_fairness_perf --seeds <LIST OF SEEDS> -c 36:36:36 -a 0.0 -b 1.0 -n <NUM THROUGHPUTS>
```
