import os
import numpy as np
import re

def get_logfile_paths_helper(directory_name):
    logfile_paths = []
    for root, _, file_names in os.walk(directory_name):
        if len(file_names) > 0:
            logfile_paths.extend(
                [os.path.join(root, file_name)
                 for file_name in file_names])
    return logfile_paths

def get_logfile_paths(directory_name):
    logfile_paths = []
    for logfile_path in get_logfile_paths_helper(directory_name):
        m = re.match(
            r'.*vc=(.*)/v100=(\d+)\.p100=(\d+)\.k80=(\d+)/(.*)/seed=(\d+).log', logfile_path)
        if m is None: continue
        vc = m.group(1)
        v100s = int(m.group(2))
        p100s = int(m.group(3))
        k80s = int(m.group(4))
        policy = m.group(5)
        seed = int(m.group(6))
        logfile_paths.append((vc, v100s, p100s, k80s, policy, seed,
                              logfile_path))
    return logfile_paths

def average_jct_fn(logfile_path, min_job_id=None, max_job_id=None):
    job_completion_times = []
    with open(logfile_path, 'r') as f:
        lines = f.readlines()
        for line in lines[-10000:]:
            m = re.match(r'Average job completion time: (\d+\.\d+) seconds', line)
            if m is not None:
                return float(m.group(1)) / 3600.0
    return None


if __name__ == '__main__':
    policies = ["fifo",
                "fifo_perf",
                "fifo_packed",
                "max_min_fairness",
                "max_min_fairness_perf",
                "max_min_fairness_packed"]
    logfile_paths = sorted(get_logfile_paths(
        "/mnt1/deepak/gpusched/scheduler/logs/philly_sweep/policies"))

    for (vc, v100s, p100s, k80s, policy, seed, logfile_path) in logfile_paths:
        print(",".join([str(x) for x in
                        [vc, v100s, p100s, k80s, policy, seed, average_jct_fn(logfile_path)]]))
