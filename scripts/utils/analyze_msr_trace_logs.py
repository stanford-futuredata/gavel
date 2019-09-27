import argparse
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
    parser = argparse.ArgumentParser(
            description='Analyze Philly traces')

    parser.add_argument('-l', '--logdir-path', type=str, default='logs',
                        help='Log directory')
    args = parser.parse_args()

    logfile_paths = sorted(get_logfile_paths(args.logdir_path))
    results = {}
    for (vc, v100s, p100s, k80s, policy, seed, logfile_path) in logfile_paths:
        key = (vc, v100s, p100s, k80s, policy)
        if key not in results: results[key] = []
        results[key].append(average_jct_fn(logfile_path))
        print(",".join([str(x) for x in
                        [vc, v100s, p100s, k80s, policy, seed, average_jct_fn(logfile_path)]]))
    for key in results:
        try:
            print(key, np.mean(results[key]))
        except:
            continue
