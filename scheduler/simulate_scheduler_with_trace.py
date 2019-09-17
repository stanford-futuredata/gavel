import argparse
import datetime
import queue
import time
import datetime

import job
import policies
import scheduler
import utils


def main(args):
    jobs, arrival_times = utils.parse_trace(args.trace_file, args.run_dir)
    policy = utils.get_policy(args.policy, args.seed)

    sched = scheduler.Scheduler(policy,
                                throughputs_file=args.throughputs_file,
                                simulate=True,
                                seed=args.seed,
                                profiling_percentage=args.profiling_percentage)

    num_gpus = args.cluster_spec.split(':')
    cluster_spec = {
            'v100': int(num_gpus[0]),
            'p100': int(num_gpus[1]),
            'k80': int(num_gpus[2]),
        }
    sched.simulate(cluster_spec, arrival_times, jobs,
                   debug=args.debug,
                   checkpoint_threshold=args.checkpoint_threshold,
                   checkpoint_file=args.checkpoint_file)
    sched.get_average_jct()
    sched.get_cluster_utilization()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=utils.get_available_policies(),
                        help='Scheduler policy')
    parser.add_argument('--throughputs_file', type=str,
                        default='oracle_throughputs.json',
                        help='Oracle throughputs file')
    parser.add_argument('-c', '--cluster_spec', type=str, default='25:0:0',
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--run_dir', type=str, default='/tmp',
                        help='Run directory')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Debug')
    parser.add_argument('--checkpoint_threshold', type=int, default=None,
                        help='Create checkpoint when this job ID comes in')
    parser.add_argument('--checkpoint_file', default=None,
                        help=('Load checkpoint located at passed in'
                              'checkpoint_file'))
    parser.add_argument('--profiling_percentage', type=float, default=0.0,
                        help=('Percentage of machines dedicated to profiling '
                              'co-located job pairs'))
    main(parser.parse_args())
