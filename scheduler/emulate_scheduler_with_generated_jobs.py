import argparse
import datetime
import contextlib
import sys

import job
from job_id_pair import JobIdPair
import policies
import scheduler
import utils

def emulate(policy_name, schedule_in_rounds, throughputs_file, cluster_spec,
            lam, seed, interval, jobs_to_complete, fixed_job_duration):
    policy = utils.get_policy(policy_name, seed=seed)
    sched = scheduler.Scheduler(
                    policy,
                    schedule_in_rounds=schedule_in_rounds,
                    throughputs_file=throughputs_file,
                    seed=seed,
                    time_per_iteration=interval,
                    emulate=True)

    cluster_spec_str = 'v100:%d|p100:%d|k80:%d' % (cluster_spec['v100'],
                                                   cluster_spec['p100'],
                                                   cluster_spec['k80'])
    current_time = datetime.datetime.now()
    print('[%s] Configuration: cluster_spec=%s, policy=%s, '
           'seed=%d, lam=%f' % (current_time, cluster_spec_str, policy.name,
                                seed, lam),
          file=sys.stderr)

    sched.emulate(cluster_spec, lam=lam,
                  jobs_to_complete=jobs_to_complete,
                  fixed_job_duration=fixed_job_duration)
    average_jct = sched.get_average_jct(jobs_to_complete)
    utilization = sched.get_cluster_utilization()
    
    current_time = datetime.datetime.now()
    print('[%s] Results: average JCT=%f, utilization=%f' % (current_time,
                                                            average_jct,
                                                            utilization),
          file=sys.stderr)

def main(args):
    schedule_in_rounds = True
    throughputs_file = 'combined_throughputs.json'
    num_gpus = args.cluster_spec.split('|')
    cluster_spec = {
            'v100': int(num_gpus[0]),
            'p100': int(num_gpus[1]),
            'k80': int(num_gpus[2]),
        }

    jobs_to_complete = set()
    for i in range(args.window_start, args.window_end):
        jobs_to_complete.add(JobIdPair(i, None))

    if args.verbose:
        emulate(args.policy, schedule_in_rounds, throughputs_file,
                cluster_spec, args.lam, args.seed,
                args.interval, jobs_to_complete,
                args.fixed_job_duration)
    
    else:
        with open('/dev/null', 'w') as f:
            with contextlib.redirect_stdout(f):
                emulate(args.policy, schedule_in_rounds, throughputs_file,
                        cluster_spec, args.lam, args.seed,
                        args.interval, jobs_to_complete,
                        args.fixed_job_duration)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Emulate scheduler with generated jobs')

    parser.add_argument('-g', '--gpus', type=int, default=25,
                        help='Number of v100 GPUs')
    parser.add_argument('-c', '--cluster_spec', type=str, default='25|0|0',
                        help=('Cluster specification in the form of'
                              '#v100s|#p100s|#k80s'))
    parser.add_argument('-s', '--window-start', type=int, default=0,
                        help='Measurement window start (job ID)')
    parser.add_argument('-e', '--window-end', type=int, default=5000,
                        help='Measurement window end (job ID)')
    parser.add_argument('-p', '--policy', type=str, required=True,
                        choices=['fifo', 'fifo_perf', 'fifo_packed',
                                 'max_min_fairness', 'max_min_fairness_perf',
                                 'max_min_fairness_packed'],
                        help='Policy')
    parser.add_argument('-d', '--seed', type=int, nargs='+',
                        default=0, help='Random seed')
    parser.add_argument('-i', '--interval', type=int, default=1920,
                        help='Interval length (in seconds)')
    parser.add_argument('-l', '--lam', type=float, required=True,
                        help='Lambda for Poisson arrival rate')
    parser.add_argument('-f', '--fixed-job-duration', type=int, default=None,
                        help=('If set, fixes the duration of all jobs to the '
                              'specified value (in seconds)'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose')
    args = parser.parse_args()
    main(args)

