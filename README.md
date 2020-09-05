# Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads

This repository contains the source code implementation of the OSDI paper
"Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads".

## Directory Structure

### `scheduler`
Code for the scheduler, including the scheduling mechanism and simulator
(`scheduler.py`), implementations of performance-aware policies (`policies/`),
`GavelIterator` as a Python module, and a communication stack between the scheduler
and workers that uses [gRPC](https://grpc.io/) (`runtime/`).

`scheduler/notebooks` contains parsing and plotting code to analyze experiment
runs.

### `workloads`
Implementations of target workloads in PyTorch, including changes needed to
integrate with the `GavelIterator`.


## Setup

### Software Dependencies

Gavel is implemented in Python. We have tested Gavel on Ubuntu 16.04 with Python 3.8.
Python 3.8 can be installed using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Required software dependencies can be installed using,

```bash
apt-get -y install cmake g++ gcc libnuma-dev make numactl zlib1g-dev
pip install -r scheduler/requirements.txt
cd scheduler; make
```

These software dependencies have already been installed on the following
AMI on Amazon EC2,

| Field  | Value |
| -------------  | ------------- |
| Cloud Provider | AWS |
| Region         | us-east-1  |
| AMI ID         | ami-03e41a79bb745ce18  |
| AMI Name       | gavel |

See [this link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami.html)
for how to find and launch a public AMI (this assumes you have a valid billable AWS account setup).

## Getting Started

Gavel's heterogeneity-aware policies and scheduling mechanism can be evaluated
either in simulation or on a physical cluster.

To evaluate variants of the LAS policy (`max_min_fairness*`) in
simulation, one can use the following command line (this sweep script runs
the different policies for multiple _continuous_ traces, generated using
different seeds and Poisson arrival rates):

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 6 -p max_min_fairness max_min_fairness_perf --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 1.0 -n 5
```

Other arguments for the `run_sweep_continuous.py` script are
shown using the `-h` option:

```bash
usage: run_sweep_continuous.py [-h] [-l LOG_DIR] [-s WINDOW_START] [-e WINDOW_END] [-t TIMEOUT] [-j PROCESSES] [-p POLICIES [POLICIES ...]] [-c CLUSTER_SPEC [CLUSTER_SPEC ...]]
                               [--num_gpus_per_server NUM_GPUS_PER_SERVER] [--seeds SEEDS [SEEDS ...]] [-i INTERVAL] [-f FIXED_JOB_DURATION]
                               [--cutoff-throughputs-file CUTOFF_THROUGHPUTS_FILE] [--throughputs-file THROUGHPUTS_FILE] [-m] [--generate-multi-priority-jobs]
                               [--simulate-steady-state] [--solver {ECOS,GUROBI,SCS}] [-v] [--checkpoint-threshold CHECKPOINT_THRESHOLD]
                               [--profiling_percentages PROFILING_PERCENTAGES [PROFILING_PERCENTAGES ...]] [--num_reference_models NUM_REFERENCE_MODELS [NUM_REFERENCE_MODELS ...]]
                               [--ideal] [-a THROUGHPUT_LOWER_BOUND] [-b THROUGHPUT_UPPER_BOUND] [-n NUM_DATA_POINTS] [-u UTILIZATION_THRESHOLD]

Sweep through lambda values

optional arguments:
  -h, --help            show this help message and exit
  -l LOG_DIR, --log-dir LOG_DIR
                        Log directory
  -s WINDOW_START, --window-start WINDOW_START
                        Measurement window start (job ID)
  -e WINDOW_END, --window-end WINDOW_END
                        Measurement window end (job ID)
  -t TIMEOUT, --timeout TIMEOUT
                        Timeout (in seconds) for each run
  -j PROCESSES, --processes PROCESSES
                        Number of processes to use in pool (use as many as available if not specified)
  -p POLICIES [POLICIES ...], --policies POLICIES [POLICIES ...]
                        List of policies to sweep
  -c CLUSTER_SPEC [CLUSTER_SPEC ...], --cluster-spec CLUSTER_SPEC [CLUSTER_SPEC ...]
                        Cluster specification in the form of #v100s:#p100s:#k80s
  --num_gpus_per_server NUM_GPUS_PER_SERVER
                        Cluster specification in the form of #v100s:#p100s:#k80s
  --seeds SEEDS [SEEDS ...]
                        List of random seeds
  -i INTERVAL, --interval INTERVAL
                        Interval length (in seconds)
  -f FIXED_JOB_DURATION, --fixed-job-duration FIXED_JOB_DURATION
                        If set, fixes the duration of all jobs to the specified value (in seconds)
  --cutoff-throughputs-file CUTOFF_THROUGHPUTS_FILE
                        If set, uses the attached cutoff_throughputs JSON file in sweep to limit args run
  --throughputs-file THROUGHPUTS_FILE
                        Oracle throughputs file
  -m, --generate-multi-gpu-jobs
                        If set, generates multi-GPU jobs according to a pre-defined distribution
  --generate-multi-priority-jobs
                        If set, generates some jobs with higher priority
  --simulate-steady-state
                        If set, adds as many jobs as there are workers before beginning the simulation.
  --solver {ECOS,GUROBI,SCS}
                        CVXPY solver
  -v, --verbose         Verbose
  --checkpoint-threshold CHECKPOINT_THRESHOLD
                        Checkpoint threshold, None if checkpointing is disabled. Checkpoint is created after this job ID is added.
  --profiling_percentages PROFILING_PERCENTAGES [PROFILING_PERCENTAGES ...]
                        Percentages of machines dedicated to profiling co-located job pairs
  --num_reference_models NUM_REFERENCE_MODELS [NUM_REFERENCE_MODELS ...]
                        Number of reference models to use when estimating throughputs
  --ideal               Run allocations 100% ideally

Automatic sweep:
  -u UTILIZATION_THRESHOLD, --utilization-threshold UTILIZATION_THRESHOLD
                        Utilization threshold to use when automatically sweeping lambdas

Sweep over fixed range:
  -a THROUGHPUT_LOWER_BOUND, --throughput-lower-bound THROUGHPUT_LOWER_BOUND
                        Lower bound for throughput interval to sweep
  -b THROUGHPUT_UPPER_BOUND, --throughput-upper-bound THROUGHPUT_UPPER_BOUND
                        Upper bound for throughput interval to sweep
  -n NUM_DATA_POINTS, --num-data-points NUM_DATA_POINTS
                        Number of data points to sweep through
```

To evaluate policies on static traces (jobs only added to the cluster at the start
of the trace), one can use the `scripts/sweeps/run_sweep_static.py` script, which
runs different policies on multiple _static_ traces, generated using different
seeds and numbers of jobs.

For more detailed instructions on how to reproduce results from the OSDI paper,
see [EXPERIMENTS.md](EXPERIMENTS.md).
