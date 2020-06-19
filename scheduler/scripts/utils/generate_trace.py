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

def get_total_steps(rng, durations, throughputs, job_type, scale_factor):
    duration = rng.choice(durations) * 3600
    steps =\
        int(throughputs['v100'][(job_type, scale_factor)]['null'] * duration)
    return max(1, steps)

def main(args):
    job_generator = random.Random()
    job_generator.seed(args.seed)

    interarrival_time_generator = random.Random()
    interarrival_time_generator.seed(args.seed + 1)

    duration_generator = random.Random()
    duration_generator.seed(args.seed + 2)
    
    scale_factor_generator = random.Random()
    scale_factor_generator.seed(args.seed + 3)

    throughputs = utils.read_all_throughputs_json_v2(args.throughputs_file)

    durations = np.linspace(args.min_duration, args.max_duration,
                            args.num_durations)

    prev_arrival_time = None
    with open(args.output_file, 'w') as f:
        for i in range(args.num_jobs):
            job_template = job_generator.choice(JobTable)
            job_type = job_template.model
            command = job_template.command
            num_steps_arg = job_template.num_steps_arg
            needs_data_dir = job_template.needs_data_dir
            scale_factor = 1
            if args.generate_multi_gpu_jobs and job_template.distributed:
                r = scale_factor_generator.uniform(0, 1)
                if 0.7 <= r <= 0.8:
                    scale_factor = 2
                elif 0.8 <= r:
                    scale_factor = 4
            total_steps = get_total_steps(duration_generator,
                                          durations,
                                          throughputs,
                                          job_type,
                                          scale_factor)
            priority_weight = 1
            SLO = -1
            if prev_arrival_time is None:
                arrival_time = 0
            elif args.lam > 0:
                arrival_time = (prev_arrival_time +
                                generate_interarrival_time(
                                    interarrival_time_generator, args.lam))
            prev_arrival_time = arrival_time
            job = Job(job_id=None,
                      job_type=job_type,
                      command=command,
                      num_steps_arg=num_steps_arg,
                      total_steps=total_steps,
                      duration=None,
                      scale_factor=scale_factor,
                      priority_weight=priority_weight,
                      SLO=SLO,
                      needs_data_dir=needs_data_dir)
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
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file name')
    args = parser.parse_args()
    main(args)
