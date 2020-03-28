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
    jobs, arrival_times = utils.parse_trace(args.trace_file, args.run_dir)
    job_queue = queue.Queue()
    for (job, arrival_time) in zip(jobs, arrival_times):
        job_queue.put((job, arrival_time))
    policy = utils.get_policy(args.policy, solver=args.solver, seed=args.seed)
    sched = scheduler.Scheduler(policy,
                                seed=args.seed)
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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=utils.get_available_policies(),
                        help='Scheduler policy')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI'],
                        default='ECOS', help='CVXPY solver')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Directory to run jobs from')
    main(parser.parse_args())
