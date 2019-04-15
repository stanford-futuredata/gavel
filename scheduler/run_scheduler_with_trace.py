import argparse
import datetime
import queue

import job
import scheduler

def parse_trace(trace_file):
    jobs = []
    with open(trace_file, 'r') as f:
        for line in f:
            command, num_steps_arg, total_steps = line.split('\t')
            jobs.append(job.Job(job_id=None,
                                job_type=None,
                                command=command,
                                num_steps_arg=num_steps_arg,
                                total_steps=total_steps,
                                duration=None))
    return jobs

def main(args):
    jobs = parse_trace(args.trace_file)
    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(queue)
    sched = scheduler.Scheduler(policy, job_packing=False)
    start_time = datetime.datetime.now()
    while not job_queue.empty():
        job = job_queue.get()
        current_time = datetime.datetime.now()
        remaining_time = start_time + current_time - job.arrival_time
        if remaining_time.seconds > 0:
            time.sleep(remaining_time.seconds)
        sched.add_job(job)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        help='Scheduler policy')
    main(parser.parse_args())
