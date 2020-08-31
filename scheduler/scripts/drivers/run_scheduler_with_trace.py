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

SLEEP_TIME = 10

def main(args):
    # Set up jobs.
    jobs_to_complete = set()
    jobs, arrival_times = utils.parse_trace(args.trace_file)
    if args.window_start is not None and args.window_end is not None:
        for i in range(args.window_start, args.window_end):
            jobs_to_complete.add(JobIdPair(i, None))
    else:
        for i in range(len(jobs)):
            jobs_to_complete.add(JobIdPair(i, None))
    job_queue = queue.Queue()
    for (job, arrival_time) in zip(jobs, arrival_times):
        job_queue.put((job, arrival_time))

    # Instantiate scheduler.
    policy = utils.get_policy(args.policy, solver=args.solver, seed=args.seed)
    sched = scheduler.Scheduler(policy,
                                seed=args.seed,
                                throughputs_file=args.throughputs_file,
                                time_per_iteration=args.time_per_iteration,
                                expected_num_workers=args.expected_num_workers,
                                max_rounds=args.max_rounds)

    try:
        # Submit jobs to the scheduler.
        start_time = datetime.datetime.now()
        while not job_queue.empty() and not sched.is_done(jobs_to_complete):
            job, arrival_time = job_queue.get()
            while True:
                current_time = datetime.datetime.now()
                elapsed_seconds = (current_time - start_time).seconds
                remaining_time = arrival_time - elapsed_seconds
                if remaining_time <= 0:
                    job_id = sched.add_job(job)
                    break
                elif sched.is_done(jobs_to_complete):
                    break
                else:
                    time.sleep(SLEEP_TIME)

        # Wait for scheduler to complete.
        sleep_seconds = 30
        while not sched.is_done(jobs_to_complete):
            time.sleep(sleep_seconds)

        # Print summary information.
        sched.get_average_jct(jobs_to_complete)
        sched.get_completed_steps(jobs_to_complete)
        sched.get_cluster_utilization()
        elapsed_time = (datetime.datetime.now() - start_time).seconds
        print('Total time taken: %d seconds' % (elapsed_time))
    except KeyboardInterrupt as e:
        pass
    finally:
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
    parser.add_argument('--time_per_iteration', type=int, default=360,
                        help='Time per iteration in seconds')
    parser.add_argument('-s', '--window-start', type=int, default=None,
                        help='measurement window start (job id)')
    parser.add_argument('-e', '--window-end', type=int, default=None,
                        help='Measurement window end (job ID)')
    parser.add_argument('--max_rounds', type=int, default=None,
                        help='Maximum number of rounds to run')
    main(parser.parse_args())
