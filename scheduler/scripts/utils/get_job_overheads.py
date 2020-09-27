import argparse
import datetime
import os
import re

def get_job_overhead(timeline):
    computation_time = 0.0
    total_time = 0.0
    current_dispatch_start_time = None
    current_compute_start_time = None
    for (timestamp, event, status, _) in timeline:
        if event == 'DISPATCHER' and status == 'LAUNCH':
            if current_dispatch_start_time is None:
                current_dispatch_start_time = timestamp
        elif event == 'INIT' and status == 'COMPLETE':
            current_compute_start_time = timestamp
        elif event == 'SAVE CHECKPOINT' and status == 'END':
            delta = (timestamp - current_compute_start_time)
            computation_time += delta.total_seconds()
            current_compute_start_time = None
        elif event == 'DISPATCHER' and status == 'COMPLETE':
            delta = (timestamp - current_dispatch_start_time)
            total_time += delta.total_seconds()
            current_dispatch_start_time = None
    if total_time == 0.0:
        computation_time = 0.0
    elif computation_time == 0.0:
        total_time = 0.0
    return (computation_time, total_time)

def parse_timeline_file(timeline_file):
    with open(timeline_file, 'r') as f:
        lines = f.read().strip().split('\n')
    timeline = []
    for line in lines:
        match = re.match('\[(.*)\] \[(.*)\] \[(.*)\]\ ?(.*)', line)
        if match is None:
            self._logger.error(
                'Malformed log line: {0}'.format(line))
            continue
        timestamp = datetime.datetime.strptime(match.group(1),
                                               '%Y-%m-%d %H:%M:%S')
        event = match.group(2)
        status = match.group(3)
        message = match.group(4)
        timeline.append((timestamp, event, status, message))
    timeline.sort(key=lambda x: x[0])
    return timeline

def main(args):
    if not os.path.isdir(args.timeline_dir):
        raise ValueError(
            'Invalid timeline directory \"{0}\"'.format(args.timeline_dir))
    for model in os.listdir(args.timeline_dir):
        model_dir = os.path.join(args.timeline_dir, model)
        for num_jobs in os.listdir(model_dir):
            num_jobs_dir = os.path.join(model_dir, num_jobs)
            print('{0}, {1}'.format(model, num_jobs))
            for i, job in enumerate(sorted(os.listdir(num_jobs_dir))):
                job_dir = os.path.join(num_jobs_dir, job)
                for j, worker in enumerate(sorted(os.listdir(job_dir))):
                    timeline_file = os.path.join(job_dir, worker)
                    timeline = parse_timeline_file(timeline_file)
                    (computation_time, total_time) = get_job_overhead(timeline)
                    overhead = (total_time - computation_time) / total_time
                    print('Job {0}, worker {1}: computation time={2:.2f}, '
                          'total time={3:.2f}, overhead={4:.2f}%'.format(
                            i, j, computation_time, total_time,
                            100.0 * overhead))
            print()

if __name__=='__main__':
    description = """
Gets overhead from timelines.

This requires the following directory structure, which is used
by scripts/microbenchmarks/sweep_models_for_overhead.py:
|-> timelines_dir
  |-> model=modelA
    |-> num_jobs=1
      |-> job_id=0
        |-> worker0.log

This is also limited to measuring the overhead for jobs running
on a single GPU, as multi-GPU configurations enable job launch latency
hiding which we do not include in this analysis."""
    parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--timeline_dir', required=True, type=str,
                        help='Root timeline directory')
    args = parser.parse_args()
    main(args)
