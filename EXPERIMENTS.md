## Setup

### Software Dependencies

Gavel is implemented in Python. We have tested Gavel with Python 3.8; this can
be installed using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Required software dependencies can be installed using,

```bash
apt-get -y install gcc g++ libnuma-dev make
pip install -r scheduler/requirements.txt
cd scheduler; make
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
