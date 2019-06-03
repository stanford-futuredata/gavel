import argparse
import datetime
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time

class Job:
    def __init__(self, model, command, num_steps):
        self._model = model
        self._command = command
        self._num_steps = num_steps

    @property
    def model(self):
        return self._model

    @property
    def command(self):
        return self._command

    @property
    def num_steps(self):
        return self._num_steps

    @num_steps.setter
    def num_steps(self, num_steps):
        self._num_steps = num_steps

job_table = [
    Job(model='ResNet-18',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'image_classification/cifar10 && python3 '
                 'main.py --data_dir=/home/keshavsanthanam/data/cifar10 '
                 '--num_steps'),
        num_steps=100),
    Job(model='Transformer',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'translation && python3 train.py -data '
                 '/home/keshavsanthanam/data/translation/multi30k.atok.low.pt '
                 '-proj_share_weight -step'),
        num_steps=100),
    Job(model='ResNet-50',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'image_classification/imagenet && python3 '
                 'main.py -j 4 -a resnet50 -b 64 /home/deepakn94/imagenet/ '
                 '--num_minibatches'),
        num_steps=25),
    Job(model='A3C',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/rl && '
                 'python3 main.py --env PongDeterministic-v4 --workers 4 '
                 '--amsgrad True --max-steps'),
        num_steps=70),
    Job(model='LM',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'language_modeling && python main.py --cuda --data '
                 '/home/keshavsanthanam/data/wikitext-2 --steps'),
        num_steps=300),
    Job(model='Recommendation',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'recommendation/scripts/ml-20m && python3 train.py -n'),
        num_steps=5),
    Job(model='CycleGAN',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'cyclegan && python3 cyclegan.py --dataset_path '
                 '/home/keshavsanthanam/data/monet2photo --decay_epoch 0 '
                 '--n_steps'),
        num_steps=1000),
]

"""
job_table = [
    Job(model='ResNet-18',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'image_classification/cifar10 && python3 '
                 'main.py --data_dir=/home/keshavsanthanam/data/cifar10 '
                 '--num_steps'),
        num_steps=4000),
    Job(model='ResNet-50',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'image_classification/imagenet && python3 '
                 'main.py -j 4 -a resnet50 -b 64 /home/deepakn94/imagenet/ '
                 '--num_minibatches'),
        num_steps=300),
    Job(model='A3C',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/rl && '
                 'python3 main.py --env PongDeterministic-v4 --workers 4 '
                 '--amsgrad True --max-steps'),
        num_steps=1000),
    Job(model='LM',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'language_modeling && python main.py --cuda --data '
                 '/home/keshavsanthanam/data/wikitext-2 --steps'),
        num_steps=1000),
    Job(model='Recommendation',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'recommendation/scripts/ml-20m && python3 train.py -n'),
        num_steps=100),
    Job(model='Transformer',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'translation && python3 train.py -data '
                 '/home/keshavsanthanam/data/translation/multi30k.atok.low.pt '
                 '-proj_share_weight -step'),
        num_steps=1000),
    Job(model='CycleGAN',
        command=('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
                 'cyclegan && python3 cyclegan.py --dataset_path '
                 '/home/keshavsanthanam/data/monet2photo --decay_epoch 0 '
                 '--n_steps'),
        num_steps=1000),
]
"""


def enable_mps():
    print('Enabling CUDA MPS')
    command = 'nvidia-cuda-mps-control -d'
    try:
        subprocess.run(command,
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,
                       shell=True)
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.stdout.decode('utf-8'))


def run_job(job):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    command = ('%s %d '
               '--throughput_estimation_interval %d') % (job.command,
                                                         job.num_steps,
                                                         job.num_steps // 100)
    try:
        output = subprocess.run(command,
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                shell=True,
                                check=True).stdout.decode('utf-8')
        return output
    except Exception as e:
        print(e)
        print(e.stdout.decode('utf-8'))
        return None

def get_throughputs(outputs):
    earliest_end_time = None
    for output in outputs:
        lines = output.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            if '[THROUGHPUT_ESTIMATION]' in lines[i]:
                _, time, _ = lines[i].split('\t')
                if (earliest_end_time is None or
                    float(time) < earliest_end_time):
                    earliest_end_time = float(time)
                break

    throughputs = None
    for output in outputs:
        lines = output.split('\n')
        start_time = None
        for line in lines:
            if '[THROUGHPUT_ESTIMATION]' in line:
                _, time, steps = line.split('\t')
                if start_time is None:
                    start_time = float(time)
                elif float(time) > earliest_end_time:
                    break
        if start_time is None:
            return (-1, -1)
        throughput = int(steps) / (float(time) - start_time)
        if throughputs is None:
            throughputs = (throughput,)
        else:
            throughputs += (throughput,)

    return throughputs

def main(args):
    global job_table

    if args.enable_mps:
        enable_mps()

    throughputs = {}
    # Get isolated throughputs.
    if args.measure_isolated_throughputs:
        for i in range(len(job_table)):
          print('%s] Running %s' % (datetime.datetime.now(),
                                    job_table[i].model))
          start_time = time.time()
          output = run_job(job_table[i])
          if not output:
              sys.exit(-1)
          runtime = time.time() - start_time
          throughput = get_throughputs([output])[0]
          # Update the number of steps the job runs for.
          old_steps = job_table[i].num_steps
          new_steps = int(old_steps *
                          args.seconds_per_measurement / runtime)
          job_table[i].num_steps = new_steps
          print('%s] %s steps %d -> %d' % (datetime.datetime.now(),
                                           job_table[i].model,
                                           old_steps,
                                           new_steps))
          throughputs[job_table[i].model] = {}
          throughputs[job_table[i].model][None] = throughput
          print('%s: %.3f (%.3f seconds)\n' % (job_table[i].model, throughput,
                                               runtime))
    # Get co-located throughputs.
    for i in range(len(job_table)):
        for j in range(len(job_table)):
            job1 = job_table[i]
            job2 = job_table[j]
            if ((job1.model in throughputs and
                 job2.model in throughputs[job1.model]) or
                (job2.model in throughputs and
                 job1.model in throughputs[job2.model])):
                continue
            if job1.model not in throughputs:
                throughputs[job1.model] = {}
            if job2.model not in throughputs:
                throughputs[job2.model] = {}
            print('%s] Running %s with %s' % (datetime.datetime.now(),
                                              job1.model, job2.model))
            outputs = []
            results = []
            pipe_list = []
            pool = multiprocessing.Pool(2)
            for job in [job1, job2]:
                results.append(pool.map_async(run_job, (job,)))
            pool.close()
            pool.join()

            success = True
            for result in results:
                output = result.get()[0]
                if output is not None:
                    outputs.append(output)
                else:
                    success = False
                    break
            key = '(%s, %s)' % (job1.model, job2.model)
            if success:
                throughput = get_throughputs(outputs)
            else:
                throughput = (-1, -1)
            throughputs[job1.model][job2.model] = throughput
            throughputs[job2.model][job1.model] = throughput[::-1]

            print('(%s, %s): (%.3f, %.3f)\n' % (job1.model, job2.model,
                                                throughput[0],
                                                throughput[1]))
    with open(args.output_file, 'w') as f:
        f.write(json.dumps(throughputs, indent=4))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Measure colocated throughput')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='Output file')
    parser.add_argument('-s', '--seconds_per_measurement', type=int,
                        default=300,
                        help='Seconds per measurement in seconds')
    parser.add_argument('-m', '--enable_mps', action='store_true',
                        default=False, help='Enable CUDA MPS')
    parser.add_argument('-i', '--measure_isolated_throughputs',
                        action='store_true', default=False,
                        help='Measure isolated throughputs')
    args = parser.parse_args()
    main(args)
