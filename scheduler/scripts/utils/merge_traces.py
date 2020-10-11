import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import datetime
import re

import utils

DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

def main(args):
    completed_jobs = set()
    offset = 0
    if args.log is not None:
        start_timestamp = None
        end_timestamp = None
        with open(args.log, 'r') as f:
            for line in f:
                if 'rpc' in line:
                    continue
                match = re.search('\[(.*?)\]', line)
                if match is None:
                    break
                timestamp = datetime.datetime.strptime(match.group(1),
                                                       DATE_FORMAT)
                if start_timestamp is None:
                    start_timestamp = timestamp
                end_timestamp = timestamp
                match = re.search('\[Job succeeded\].*Job ID: (\d+)', line)
                if match is not None:
                    job_id = int(match.group(1))
                    completed_jobs.add(job_id)
                    continue
                match = re.search('\[Job failed\].*Job ID: (\d+)', line)
                if match is not None:
                    job_id = int(match.group(1))
                    print('Job {0} has completed already'.format(job_id))
                    completed_jobs.add(job_id)
                    continue
        offset = (end_timestamp - start_timestamp).total_seconds()

    jobs_and_arrival_times = []

    jobs, arrival_times = utils.parse_trace(args.t2)
    for job, arrival_time in zip(jobs, arrival_times):
        jobs_and_arrival_times.append((job, arrival_time))

    jobs, arrival_times = utils.parse_trace(args.t1)
    for job, arrival_time in zip(jobs, arrival_times):
        if job in completed_jobs:
            continue
        elif arrival_time < offset:
            continue
        arrival_time -= offset
        jobs_and_arrival_times.append((job, arrival_time))

    with open(args.output, 'w') as f:
        for (job, arrival_time) in jobs_and_arrival_times:
            f.write('{0}\t{1}\n'.format(job, arrival_time))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Merge two traces')
    parser.add_argument('-t1', type=str, required=True, help='First trace')
    parser.add_argument('-t2', type=str, required=True, help='Second trace')
    parser.add_argument('--log', type=str, default=None,
                        help='Execution log before failure')
    parser.add_argument('--output', type=str, required=True,
                        help='Output trace')
    args = parser.parse_args()
    main(args)
