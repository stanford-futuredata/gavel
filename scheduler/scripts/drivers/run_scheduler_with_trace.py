import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

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
    jobs, arrival_times = utils.parse_trace(args.trace_file)
    job_queue = queue.Queue()
    for (job, arrival_time) in zip(jobs, arrival_times):
        job_queue.put((job, arrival_time))
    policy = utils.get_policy(args.policy, solver=args.solver, seed=args.seed)
    sched = scheduler.Scheduler(policy,
                                seed=args.seed,
                                throughputs_file=args.throughputs_file,
                                time_per_iteration=args.time_per_iteration,
                                expected_num_workers=args.expected_num_workers)
    start_time = datetime.datetime.now()
    while not job_queue.empty():
        job, arrival_time = job_queue.get()
        current_time = datetime.datetime.now()
        elapsed_seconds = (current_time - start_time).seconds
        remaining_time = arrival_time - elapsed_seconds
        if remaining_time > 0:
            time.sleep(remaining_time)
        job_id = sched.add_job(job)

    sleep_seconds = 30
    while not sched.is_done():
        time.sleep(sleep_seconds)

    print("Total time taken: %d seconds" % (datetime.datetime.now() - start_time).seconds)
    sched.get_average_jct()
    sched.get_cluster_utilization()
    sched.shutdown()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=utils.get_available_policies(),
                        help='Scheduler policy')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI', 'SCS'],
                        default='ECOS', help='CVXPY solver')
    parser.add_argument('--throughputs_file', type=str,
                        default=None,
                        help='Oracle throughputs file')
    parser.add_argument('--expected_num_workers', type=int, default=None,
                        help='Total number of workers expected')
    parser.add_argument('--time_per_iteration', type=int, default=1920,
                        help='Time per iteration in seconds')
    main(parser.parse_args())
