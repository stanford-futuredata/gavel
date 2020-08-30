import os
import random
import re

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import sys; sys.path.append("../../..")
from job_table import JobTable


def get_logfile_paths_helper(directory_name):
    logfile_paths = []
    for root, _, file_names in os.walk(directory_name):
        if len(file_names) > 0:
            logfile_paths.extend(
                [os.path.join(root, file_name)
                 for file_name in file_names])
    return logfile_paths

def get_logfile_paths(directory_name, static_trace=False):
    logfile_paths = []
    for logfile_path in get_logfile_paths_helper(directory_name):
        if static_trace:
            m = re.match(
                r'.*v100=(\d+)\.p100=(\d+)\.k80=(\d+)/(.*)/seed=(\d+)/'
                 'num_total_jobs=(\d+)\.log', logfile_path)
        else:
            m = re.match(
                r'.*v100=(\d+)\.p100=(\d+)\.k80=(\d+)/(.*)/seed=(\d+)/'
                 'lambda=(\d+\.\d+)\.log', logfile_path)
        if m is None: continue
        v100s = int(m.group(1))
        p100s = int(m.group(2))
        k80s = int(m.group(3))
        policy = m.group(4)
        seed = int(m.group(5))
        lambda_or_num_total_jobs = float(m.group(6))
        logfile_paths.append((v100s, p100s, k80s, policy, seed,
                              lambda_or_num_total_jobs, logfile_path))
    return logfile_paths

def prune(logfile_paths, v100s, p100s, k80s, policy, seed=None):
    if seed is None:
        return sorted([(x[5], x[6], x[4]) for x in logfile_paths
                       if x[0] == v100s and x[1] == p100s and
                       x[2] == k80s and x[3] == policy])
    else:
        return sorted([(x[5], x[6]) for x in logfile_paths
                       if x[0] == v100s and x[1] == p100s and
                       x[2] == k80s and x[3] == policy and
                       x[4] == seed])

def average_jct_fn(logfile_path, min_job_id=None, max_job_id=None):
    job_completion_times = []
    with open(logfile_path, 'r') as f:
        lines = f.readlines()
        for line in lines[-10000:]:
            m = re.match(r'Job (\d+): (\d+\.\d+)', line)
            if m is not None:
                job_id = int(m.group(1))
                job_completion_time = float(m.group(2))
                if min_job_id is None or min_job_id <= job_id:
                    if max_job_id is None or job_id <= max_job_id:
                        job_completion_times.append(
                            job_completion_time)
    if len(job_completion_times) == 0:
        return 110.0
    return np.mean(job_completion_times) / 3600

def average_jct_low_priority_fn(logfile_path, min_job_id=None,
                                max_job_id=None):
    job_completion_times = []
    with open(logfile_path, 'rb') as f:
        f.seek(-8192, os.SEEK_END)
        text = f.read().decode('utf-8')
        lines = text.split('\n')
        for line in lines[-5:]:
            m = re.match(r'Average job completion time \(low priority\): (\d+\.\d+) seconds', line)
            if m is not None:
                return float(m.group(1)) / 3600
    return None

def average_jct_high_priority_fn(logfile_path, min_job_id=None,
                                 max_job_id=None):
    job_completion_times = []
    with open(logfile_path, 'rb') as f:
        f.seek(-8192, os.SEEK_END)
        text = f.read().decode('utf-8')
        lines = text.split('\n')
        for line in lines[-5:]:
            m = re.match(r'Average job completion time \(high priority\): (\d+\.\d+) seconds', line)
            if m is not None:
                return float(m.group(1)) / 3600
    return None

def makespan_fn(logfile_path):
    job_completion_times = []
    with open(logfile_path, 'r') as f:
        lines = f.readlines()
        for line in lines[-10000:]:
            m = re.match(r'Total duration: (\d+\.\d+) seconds', line)
            if m is not None:
                makespan = float(m.group(1)) / 3600.
                return makespan
    return None

def get_job_durations(seed, generate_multigpu_jobs):
    job_generator = random.Random()
    job_generator.seed(seed+2)
    
    job_durations = []
    for i in range(5000):
        r = job_generator.uniform(0, 1)
        scale_factor = 1
        if 0.7 <= r <= 0.8:
            scale_factor = 2
        elif 0.8 <= r <= 0.95:
            scale_factor = 4
        elif 0.95 <= r:
            scale_factor = 8
        if not generate_multigpu_jobs:
            scale_factor = 1
        if job_generator.random() >= 0.8:
            job_duration = 60 * (10 ** job_generator.uniform(3, 4))
        else:
            job_duration = 60 * (10 ** job_generator.uniform(1.5, 3))

        while True:
            job_template = job_generator.choice(JobTable)
            if (scale_factor == 1 or
                (scale_factor > 1 and job_template.distributed)):
                break
                
        job_durations.append((job_duration, job_template, scale_factor))
    return job_durations

def get_jcts(logfile_path, seed, min_job_id=None, max_job_id=None):
    job_completion_times = []
    job_durations = get_job_durations(seed, generate_multigpu_jobs=True)
    with open(logfile_path, 'r') as f:
        lines = f.readlines()
        for line in lines[-10000:]:
            m = re.match(r'Job (\d+): (\d+\.\d+)', line)
            if m is not None:
                job_id = int(m.group(1))
                job_completion_time = float(m.group(2))
                if min_job_id is None or min_job_id <= job_id:
                    if max_job_id is None or job_id <= max_job_id:
                        job_duration, job_template, scale_factor = job_durations[job_id]
                        job_completion_times.append(
                            (job_completion_time, job_duration))
    return [(x[0] / 3600.0, x[1] / 3600.0) for x in job_completion_times]
