import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import datetime
import re

import utils

DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

def main(args):
    remaining_steps = {}
    in_progress_steps = {}
    scale_factors = {}

    jobs, arrival_times = utils.parse_trace(args.input_trace)
    for i, job in enumerate(jobs):
        scale_factors[i] = job.scale_factor
        remaining_steps[i] = job.total_steps
        in_progress_steps[i] = []

    offset = 0
    if args.log is not None:
        start_timestamp = None
        end_timestamp = None
        with open(args.log, 'r') as f:
            for line in f:
                if 'rpc' not in line:
                    match = re.search('\[(.*?)\]', line)
                    if match is None:
                        break
                    timestamp = datetime.datetime.strptime(match.group(1),
                                                           DATE_FORMAT)
                    if start_timestamp is None:
                        start_timestamp = timestamp
                    end_timestamp = timestamp

                # Check if job completed a micro-task.
                match = re.search('Job (\d+) has (\d+) remaining steps', line)
                if match is not None:
                    job_id = int(match.group(1))
                    remaining_steps[job_id] = int(match.group(2))
                    in_progress_steps[job_id] = 0
                    continue

                # Check if job has any in-progress steps.
                match = re.search('Received lease update request.*'
                                  'job_id=(\d+).* steps=(\d+)', line)
                if match is not None:
                    job_id = int(match.group(1))
                    if len(in_progress_steps[job_id]) == scale_factors[job_id]:
                        in_progress_steps[job_id] = []
                    in_progress_steps[job_id].append(int(match.group(2)))
                    continue

                # Check if job succeeded.
                match = re.search('\[Job succeeded\].*Job ID: (\d+)', line)
                if match is not None:
                    job_id = int(match.group(1))
                    del remaining_steps[job_id]
                    continue

                # Check if job failed.
                match = re.search('\[Job failed\].*Job ID: (\d+)', line)
                if match is not None:
                    job_id = int(match.group(1))
                    del remaining_steps[job_id]
                    continue

                # Verify that jobs received all update requests in the round.
                match = re.search('END', line)
                if match is not None:
                    for job_id in in_progress_steps:
                        num_updates = len(in_progress_steps[job_id])
                        assert(num_updates == 0 or
                               num_updates == scale_factors[job_id])

        offset = (end_timestamp - start_timestamp).total_seconds()

    with open(args.output_trace, 'w') as f:
        for job_id, (job, arrival_time) in enumerate(zip(jobs, arrival_times)):
            if job_id not in remaining_steps:
                continue
            job.total_steps = \
                remaining_steps[job_id] - sum(in_progress_steps[job_id])
            arrival_time = max(0, arrival_time - offset)
            f.write('{0}\t{1}\n'.format(job, arrival_time))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Merge two traces')
    parser.add_argument('--input_trace', type=str, required=True,
                        help='Input trace')
    parser.add_argument('--log', type=str, default=None,
                        help='Execution log before failure')
    parser.add_argument('--output_trace', type=str, required=True,
                        help='Output trace')
    args = parser.parse_args()
    main(args)
