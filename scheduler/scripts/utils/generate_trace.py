import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import math
import numpy as np
import random

from job import Job
from job_table import JobTable
import utils

def generate_interarrival_time(rng, lam):
    return -math.log(1.0 - rng.random()) * lam

def generate_duration(durations, rng):
    return 3600 * rng.choice(durations)

def main(args):
    job_generator = random.Random()
    job_generator.seed(args.seed)

    interarrival_time_generator = random.Random()
    interarrival_time_generator.seed(args.seed + 1)

    throughputs = utils.read_all_throughputs_json_v2(args.throughputs_file)

    durations = np.linspace(args.min_duration, args.max_duration,
                            args.num_durations)
    duration_generator_func = lambda rng: generate_duration(durations, rng)

    prev_arrival_time = None
    with open(args.output_file, 'w') as f:
        for i in range(args.num_jobs):
            job = utils.generate_job(
                    throughputs=throughputs,
                    worker_type='v100',
                    rng=job_generator,
                    job_id=None,
                    fixed_job_duration=None,
                    generate_multi_gpu_jobs=args.generate_multi_gpu_jobs,
                    generate_multi_priority_jobs=args.generate_multi_priority_jobs,
                    duration_generator_func=duration_generator_func)
            if prev_arrival_time is None:
                arrival_time = 0
            elif args.lam > 0:
                interarrival_time = \
                    generate_interarrival_time(interarrival_time_generator,
                                               args.lam)
                arrival_time = prev_arrival_time + interarrival_time
            prev_arrival_time = arrival_time
            f.write('%s\t%d\n' % (str(job), arrival_time))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic trace')
    parser.add_argument('--num_jobs', type=int, required=True,
                        help='Number of jobs to generate')
    parser.add_argument('-l', '--lam', type=float, default=0.0,
                        help='Lambda for Poisson arrival rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--throughputs_file', type=str,
                        default=('simulation_throughputs.json'),
                        help='Oracle throughputs file')
    parser.add_argument('-a', '--min_duration', type=float, default=1,
                        help='Minimum job duration in hours')
    parser.add_argument('-b', '--max_duration', type=float, default=4,
                        help='Maximum job duration in hours')
    parser.add_argument('-n', '--num_durations', type=int, default=4,
                        help='Number of possible job durations')
    parser.add_argument('-m', '--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file name')
    args = parser.parse_args()
    main(args)
