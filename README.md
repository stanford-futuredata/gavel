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
