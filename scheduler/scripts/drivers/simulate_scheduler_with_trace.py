import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import datetime
import queue
import time
import datetime

import job
from job_id_pair import JobIdPair
import policies
import scheduler
import utils


def main(args):
    jobs, arrival_times = utils.parse_trace(args.trace_file)
    policy = utils.get_policy(args.policy, solver=args.solver, seed=args.seed)

    sched = scheduler.Scheduler(policy,
                                throughputs_file=args.throughputs_file,
                                simulate=True,
                                seed=args.seed,
                                time_per_iteration=args.time_per_iteration)

    num_gpus = args.cluster_spec.split(':')
    cluster_spec = {
        'v100': int(num_gpus[0]),
        'p100': int(num_gpus[1]),
        'k80': int(num_gpus[2]),
    }
    num_gpus_per_server_split = args.num_gpus_per_server.split(':')
    num_gpus_per_server = {
        'v100': int(num_gpus_per_server_split[0]),
        'p100': int(num_gpus_per_server_split[1]),
        'k80': int(num_gpus_per_server_split[2]),
    }
    if args.window_start is not None and args.window_end is not None:
        jobs_to_complete = set()
        for i in range(args.window_start, args.window_end):
            jobs_to_complete.add(JobIdPair(i, None))
    else:
        jobs_to_complete = None

    sched.simulate(cluster_spec, arrival_times, jobs,
                   debug=args.debug,
                   checkpoint_threshold=args.checkpoint_threshold,
                   checkpoint_file=args.checkpoint_file,
                   num_gpus_per_server=num_gpus_per_server,
                   jobs_to_complete=jobs_to_complete)
    sched.get_average_jct(jobs_to_complete)
    sched.get_cluster_utilization()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=utils.get_available_policies(),
                        help='Scheduler policy')
    parser.add_argument('--throughputs_file', type=str,
                        default='simulation_throughputs.json',
                        help='Oracle throughputs file')
    parser.add_argument('-c', '--cluster_spec', type=str, default='25:0:0',
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--num_gpus_per_server', type=str, default='1:1:1',
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI', 'SCS'],
                        default='ECOS', help='CVXPY solver')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Debug')
    parser.add_argument('--checkpoint_threshold', type=int, default=None,
                        help='Create checkpoint when this job ID comes in')
    parser.add_argument('--checkpoint_file', default=None,
                        help=('Load checkpoint located at passed in'
                              'checkpoint_file'))
    parser.add_argument('--time_per_iteration', type=int, default=1920,
                        help='Time per iteration in seconds')
    parser.add_argument('-s', '--window-start', type=int, default=None,
                        help='measurement window start (job id)')
    parser.add_argument('-e', '--window-end', type=int, default=None,
                        help='Measurement window end (job ID)')
    main(parser.parse_args())
