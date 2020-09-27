import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import asyncio
import numpy as np
import re
import time

from job import Job
from job_id_pair import JobIdPair
from job_table import JobTable

BASE_SCHEDULER_COMMAND = \
    """python scripts/drivers/run_scheduler_with_trace.py \
        --seed 0 \
        --solver ECOS \
        --throughputs_file physical_cluster_throughputs.json \
        --time_per_iteration 360 \
        --policy max_min_fairness \
        --expected_num_workers 1"""
BASE_WORKER_COMMAND = \
    """python worker.py -i 127.0.1.1 -s 50060 -w 50061 -g 1 -t p100"""
INFINITY = 10000000000

async def run(cmd, sleep_seconds=None):
    if sleep_seconds is not None:
        time.sleep(sleep_seconds)
    proc = \
        await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return (stdout, stderr)

def main(args):
    loop = asyncio.get_event_loop()
    max_rounds = args.max_rounds
    timelines_dir = args.timelines_dir
    trace_file = '/tmp/overhead.trace'
    models = [
        ('ResNet-18 (batch size 256)', 'resnet18'),
        ('ResNet-50 (batch size 128)', 'resnet50'),
        ('Transformer (batch size 256)', 'transformer'),
        ('LM (batch size 80)', 'lm'),
        ('Recommendation (batch size 8192)', 'recommendation'),
        ('A3C', 'a3c'),
        ('CycleGAN', 'cyclegan'),
    ]

    for (model, model_dir) in models:
        model_dir = os.path.join(timelines_dir, 'model={0}'.format(model_dir))
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        for job_template in JobTable:
            if job_template.model == model:
                break
        if job_template.model != model:
            print('Could not find model {0} in JobTable!'.format(model))
            continue

        jobs = []
        for num_jobs in [1, 2]:
            job = Job(job_id=JobIdPair(num_jobs-1, None),
                      job_type=model,
                      command=job_template.command,
                      working_directory=job_template.working_directory,
                      num_steps_arg=job_template.num_steps_arg,
                      total_steps=INFINITY,
                      duration=3600,
                      scale_factor=1,
                      priority_weight=1,
                      SLO=None,
                      needs_data_dir=job_template.needs_data_dir)

            mode = 'w' if num_jobs == 1 else 'a'
            with open(trace_file, mode) as f:
                f.write('{0}\t{1}\n'.format(str(job), 0))

            num_jobs_dir = \
                os.path.join(model_dir, 'num_jobs={0}'.format(num_jobs))
            if not os.path.isdir(num_jobs_dir):
                os.mkdir(num_jobs_dir)

            scheduler_cmd = BASE_SCHEDULER_COMMAND
            scheduler_cmd += ' --max_rounds {0}'.format(max_rounds)
            scheduler_cmd += ' --trace_file {0}'.format(trace_file)
            scheduler_cmd += ' --timelines_dir {0}'.format(num_jobs_dir)
            worker_cmd = BASE_WORKER_COMMAND
            worker_cmd += ' --run_dir {0}'.format(args.run_dir)
            worker_cmd += ' --data_dir {0}'.format(args.data_dir)
            worker_cmd += ' --checkpoint_dir {0}'.format(args.checkpoint_dir)

            print('Running {0} with {1} jobs...'.format(model, num_jobs))
            loop.run_until_complete(
                asyncio.gather(run(scheduler_cmd), run(worker_cmd, 1)))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Sweep models to collect job timelines for '
                    'measuring overhead')
    parser.add_argument('--max_rounds', type=int, default=10,
                        help='Maximum number of rounds to run')
    parser.add_argument('--timelines_dir', type=str, required=True,
                        help='Root directory to write timelines to')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Directory to run jobs from')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory where data is stored')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory where checkpoints is stored')
    args = parser.parse_args()
    main(args)
