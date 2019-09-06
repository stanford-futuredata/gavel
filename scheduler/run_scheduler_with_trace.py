import argparse
import datetime
import queue
import time
import datetime

import job
import policies
import scheduler
import utils

def parse_trace(trace_file):
    jobs = []
    with open(trace_file, 'r') as f:
        for line in f:
            job_type, command, num_steps_arg, total_steps, arrival_time, scale_factor = \
                    line.split('\t')
            jobs.append((job.Job(job_id=None,
                                 job_type=job_type,
                                 command=command,
                                 num_steps_arg=num_steps_arg,
                                 total_steps=int(total_steps),
                                 duration=None,
                                 scale_factor=int(scale_factor)),
                        int(arrival_time)))
    return jobs

def main(args):
    jobs = parse_trace(args.trace_file)
    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    policy = utils.get_policy(args.policy, args.seed)
    sched = scheduler.Scheduler(policy, simulate=False, throughputs_file="oracle_throughputs.json",
                                seed=args.seed, time_per_iteration=10)
    start_time = datetime.datetime.now()
    while not job_queue.empty():
        job, arrival_time = job_queue.get()
        current_time = datetime.datetime.now()
        elapsed_seconds = (current_time - start_time).seconds
        remaining_time = arrival_time - elapsed_seconds
        if remaining_time > 0:
            time.sleep(remaining_time)
        job_id = sched.add_job(job)

    sleep_seconds = 5
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
                        choices=['isolated', 'max_min_fairness',
                                 'max_min_fairness_packed', 'min_total_duration',
                                 'min_total_duration_packed', 'fifo'],
                        help='Scheduler policy')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    main(parser.parse_args())
