# OSDI 2020 Experiments

This document describes how to run the main experiments in the OSDI 2020 paper.
The goal of this document is to satisfy the "Artifact Functional" and
"Results Reproduced" badges.

## Setup

### Software Dependencies

Gavel is implemented in Python. We have tested Gavel's simulator on Ubuntu 16.04
with Python 3.8; this can be installed using
[Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Required software dependencies can be installed using,

```bash
apt-get -y install gcc g++ libnuma-dev make
pip install -r scheduler/requirements.txt
cd scheduler; make
```

These software dependencies have already been installed on the following
AMI on Amazon EC2,

| Field  | Value |
| -------------  | ------------- |
| Cloud Provider | AWS |
| Region         | us-east-1  |
| AMI ID         | ami-0ba07d9d617dcef04  |
| AMI Name       | gavel |

See [this link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami.html)
for how to find and launch a public AMI (this assumes you have a valid billable AWS account setup).


## Reproducing Experiments

Gavel's heterogeneity-aware policies and scheduling mechanism can be evaluated
either in simulation or on a physical cluster.

The evaluation in the paper largely shows results on a simulated cluster.

### Figure 8: Least Attained Service Policy on Continuous-Single Trace

To reproduce Figure 8 in the paper (that is, evaluate variants of the LAS
policy (`max_min_fairness*`) in simulation), one can use the following command
line (this sweep script runs the different policies for multiple traces,
generated using different seeds and Poisson arrival rates):

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l <LOG_DIRECTORY> -j <NUM_CORES> -p allox gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed --seeds <LIST OF SEEDS> -c 36:36:36 -a 0.0 -b 6.0 -n 16
```

The output of this script looks like this,

```bash
>> python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l test_logs -j 6 -p allox gandiva max_min_fairness max_min_fairness_perf --seeds 42 1234 15 -c 36:36:36 -a 0.0 -b 6.0 -n 16
[2020-09-03 17:17:49.260052] Running 180 total experiment(s)...
[2020-09-03 17:17:49.535227] [Experiment ID:  0] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=42, lam=9000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.600210] [Experiment ID:  1] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=1234, lam=9000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.699429] [Experiment ID:  2] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=15, lam=9000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.727137] [Experiment ID:  4] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=1234, lam=4500.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.826323] [Experiment ID:  3] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=42, lam=4500.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.875449] [Experiment ID:  5] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=15, lam=4500.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:21:27.936718] [Experiment ID:  3] Results: average JCT=59770.441018, utilization=0.121262
[2020-09-03 17:21:28.072957] [Experiment ID:  6] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=42, lam=3000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:21:30.507841] [Experiment ID:  4] Results: average JCT=64695.450528, utilization=0.123312
[2020-09-03 17:21:30.639389] [Experiment ID:  7] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=1234, lam=3000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:21:31.027912] [Experiment ID:  5] Results: average JCT=55072.401336, utilization=0.121365
[2020-09-03 17:21:31.161566] [Experiment ID:  8] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=15, lam=3000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:21:54.980905] [Experiment ID:  2] Results: average JCT=54834.129989, utilization=0.060839
[2020-09-03 17:22:00.203122] [Experiment ID:  1] Results: average JCT=64494.582453, utilization=0.061878
[2020-09-03 17:22:01.456350] [Experiment ID:  0] Results: average JCT=59492.106756, utilization=0.060797
...
```

Some policies might need to be run for higher input job rates as well. Our
experiments were run using seeds 0, 1, and 2; results with other seeds should
look similar. Note that this can take a while to complete: on the order of days
to a week. There are a couple of different ways to obtain results quicker:
a) run with a fewer number of seeds, b) sweep fewer input job rates (controlled
by the `-n` argument), c) run on a smaller cluster (controlled by
the `-c` argument), d) adjust the `-s` and `-e` arguments, which control the
size of the trace, and the size of the set of jobs of interest.

`scheduler/notebooks/figures/evaluation/continuous_jobs.ipynb` contains
code to parse the resulting logs and produce graphs (can be run using `jupyter
notebook`). The notebook should
use the right `log_directory`.

### Figure 9: Least Attained Service Policy on Continuous-Multiple Trace

To reproduce Figure 9, one can use the following command line:

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l <LOG_DIRECTORY> -j <NUM_CORES> -p gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed --seeds <LIST OF SEEDS> -c 36:36:36 -a 0.0 -b 3.0 -n 11 --generate-multi-gpu-jobs
```

`scheduler/notebooks/figures/evaluation/continuous_jobs_multigpu.ipynb` contains
relevant parsing and plotting code.

### Figure 10: Finish Time Fairness Policy on Continuous-Multiple Trace

To reproduce Figure 10, one can use the following command line:

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l <LOG_DIRECTORY> -j <NUM_CORES> -p finish_time_fairness finish_time_fairness_perf --seeds <LIST OF SEEDS> -c 36:36:36 -a 0.0 -b 3.0 -n 11 --generate-multi-gpu-jobs
```

### Makespan Policy on Static-Multiple Trace

To reproduce the results with the makespan policy:

```bash
python -u scripts/sweeps/run_sweep_static.py -l <LOG_DIRECTORY> -j <NUM_CORES> -p gandiva max_min_fairness min_total_duration min_total_duration_packed fifo gandiva --seeds <LIST OF SEEDS> -c 36:36:36 -a 0 -b 500 -n 6 --generate-multi-gpu-jobs
```

### Figure 11: Multi-Level Fairness Policy

The code for the simulation shown in Figure 11 is in `scheduler/notebooks/figures/evaluation/hierarchical.ipynb`.

### Figure 12: Policy Runtime Scaling

Policy runtimes can be measured using the following command (TODO: Fix this):

```bash
python scripts/microbenchmarks/sweep_policy_runtimes.py -n 32 -p max_min_fairness --num_trials 1
```

`scheduler/notebooks/figures/evaluation/policy_runtimes.ipynb` contains relevant
parsing and plotting code.

### Figure 13: Efficacy of Scheduling Mechanism

The time proportions returned by the policy can be used directly to grant jobs
times on each resource type between "reset events" -- this is a useful comparison
for our scheduling mechanism. This "ideal" scheduling mechanism can be run
for a given policy and trace by appending the `--ideal` argument to any of the
sweep commands above.

The round durations used by the scheduling mechanism can be similarly studied
by using the `-i` argument.
