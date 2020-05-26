import argparse
import datetime
import jsons
import csv
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from job_type_set import JobTypeSet


class Job:
    def __init__(self, model, command, num_steps, needs_data_dir=True):
        self._model = model
        self._command = command
        self._num_steps = num_steps
        self._needs_data_dir = needs_data_dir
        self._checkpoint_dir = self._model.replace(' ', '_').replace('(','').replace(')','')

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

    @property
    def needs_data_dir(self):
        return self._needs_data_dir

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value):
        self._checkpoint_dir = self._checkpoint_dir + value

job_table = [
    Job(model='ResNet-50 (batch size 16)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/imagenet && python '
                 'main.py -j 4 -a resnet50 -b 16 %s/data/imagenet/pytorch '
                 '--num_minibatches'),
        num_steps=500),
    Job(model='ResNet-50 (batch size 32)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/imagenet && python '
                 'main.py -j 4 -a resnet50 -b 32 %s/data/imagenet/pytorch '
                 '--num_minibatches'),
        num_steps=500),
    Job(model='ResNet-50 (batch size 64)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/imagenet && python '
                 'main.py -j 4 -a resnet50 -b 64 %s/data/imagenet/pytorch '
                 '--num_minibatches'),
        num_steps=500),
    Job(model='ResNet-50 (batch size 128)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/imagenet && python '
                 'main.py -j 4 -a resnet50 -b 128 %s/data/imagenet/pytorch '
                 '--num_minibatches'),
        num_steps=500),
    Job(model='ResNet-18 (batch size 16)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/cifar10 && python '
                 'main.py --data_dir=%s/data/cifar10 --batch_size 16 '
                 '--num_steps'),
        num_steps=5000),
    Job(model='ResNet-18 (batch size 32)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/cifar10 && python '
                 'main.py --data_dir=%s/data/cifar10 --batch_size 32 '
                 '--num_steps'),
        num_steps=5000),
    Job(model='ResNet-18 (batch size 64)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/cifar10 && python '
                 'main.py --data_dir=%s/data/cifar10 --batch_size 64 '
                 '--num_steps'),
        num_steps=5000),
    Job(model='ResNet-18 (batch size 128)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/cifar10 && python '
                 'main.py --data_dir=%s/data/cifar10 --batch_size 128 '
                 '--num_steps'),
        num_steps=5000),
    Job(model='ResNet-18 (batch size 256)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/cifar10 && python '
                 'main.py --data_dir=%s/data/cifar10 --batch_size 256 '
                 '--num_steps'),
        num_steps=5000),
    Job(model='Transformer (batch size 16)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'translation && python train.py -data '
                 '%s/data/translation/multi30k.atok.low.pt -batch_size 16 '
                 '-proj_share_weight -step'),
        num_steps=100),
    Job(model='Transformer (batch size 32)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'translation && python train.py -data '
                 '%s/data/translation/multi30k.atok.low.pt -batch_size 32 '
                 '-proj_share_weight -step'),
        num_steps=100),
    Job(model='Transformer (batch size 64)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'translation && python train.py -data '
                 '%s/data/translation/multi30k.atok.low.pt -batch_size 64 '
                 '-proj_share_weight -step'),
        num_steps=100),
    Job(model='Transformer (batch size 128)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'translation && python train.py -data '
                 '%s/data/translation/multi30k.atok.low.pt -batch_size 128 '
                 '-proj_share_weight -step'),
        num_steps=100),
    Job(model='Transformer (batch size 256)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'translation && python train.py -data '
                 '%s/data/translation/multi30k.atok.low.pt -batch_size 256 '
                 '-proj_share_weight -step'),
        num_steps=100),
    Job(model='A3C',
        command=('cd %s/gpusched/workloads/pytorch/rl && '
                 'python main.py --env PongDeterministic-v4 --workers 4 '
                 '--amsgrad True --max-steps'),
        num_steps=2000,
        needs_data_dir=False),
    Job(model='LM (batch size 5)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'language_modeling && python main.py --cuda --data %s/data/wikitext-2 '
                 '--batch_size 5 --steps'),
        num_steps=1000),
    Job(model='LM (batch size 10)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'language_modeling && python main.py --cuda --data %s/data/wikitext-2 '
		 '--batch_size 10 --steps'),
        num_steps=1000),
    Job(model='LM (batch size 20)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'language_modeling && python main.py --cuda --data %s/data/wikitext-2 '
                 ' --batch_size 20 --steps'),
        num_steps=1000),
    Job(model='LM (batch size 40)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'language_modeling && python main.py --cuda --data %s/data/wikitext-2 '
                 '--batch_size 40 --steps'),
        num_steps=1000),
    Job(model='LM (batch size 80)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'language_modeling && python main.py --cuda --data %s/data/wikitext-2 '
                 '--batch_size 80 --steps'),
        num_steps=1000),
    Job(model='Recommendation (batch size 512)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'recommendation/scripts/ml-20m && python train.py '
                 '--data_dir %s/data/ml-20m/pro_sg/ '
                 '--batch_size 512 -n '),
        num_steps=50),
    Job(model='Recommendation (batch size 1024)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'recommendation/scripts/ml-20m && python train.py '
                 '--data_dir %s/data/ml-20m/pro_sg/ '
                 '--batch_size 1024 -n '),
        num_steps=50),
    Job(model='Recommendation (batch size 2048)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'recommendation/scripts/ml-20m && python train.py '
                 '--data_dir %s/data/ml-20m/pro_sg/ '
                 '--batch_size 2048 -n '),
        num_steps=50),
    Job(model='Recommendation (batch size 4096)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'recommendation/scripts/ml-20m && python train.py '
                 '--data_dir %s/data/ml-20m/pro_sg/ '
                 '--batch_size 4096 -n '),
        num_steps=50),
    Job(model='Recommendation (batch size 8192)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'recommendation/scripts/ml-20m && python train.py '
                 '--data_dir %s/data/ml-20m/pro_sg/ '
                 '--batch_size 8192 -n '),
        num_steps=50),
    Job(model='CycleGAN',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'cyclegan && python cyclegan.py --dataset_path '
                 '%s/data/monet2photo --decay_epoch 0 '
                 '--n_steps'),
        num_steps=500),
]


job_table = [
    Job(model='ResNet-50 (batch size 16)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'image_classification/imagenet && python '
                 'main.py -j 4 -a resnet50 --synth -b 59 %s/data/imagenet/pytorch '
                 '--num_minibatches'),
        num_steps=500),
]
"""    Job(model='Transformer (batch size 128)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'translation && python train.py -data '
                 '%s/data/translation/multi30k.atok.low.pt -batch_size 98 '
                 '-proj_share_weight -step'),
        num_steps=100),
    Job(model='LM (batch size 80)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'language_modeling && python main.py --cuda --data %s/data/wikitext-2 '
                 '--batch_size 276 --steps'),
        num_steps=1000),
    Job(model='Recommendation (batch size 2948)',
        command=('cd %s/gpusched/workloads/pytorch/'
                 'recommendation/scripts/ml-20m && python train.py '
                 '--data_dir %s/data/ml-20m/pro_sg/ '
                 '--batch_size 2948 -n '),
        num_steps=50),
]"""


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


def run_job(job, run_dir, data_dir, checkpoint_dir):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    interval = max(1, job.num_steps // 100)
    if checkpoint_dir is None:
        checkpoint_dir = run_dir
    if job.needs_data_dir:
        if data_dir is None:
            parameterized_command = job.command % (run_dir, run_dir)
        else:
            parameterized_command = job.command % (run_dir, data_dir)
    else:
        parameterized_command = job.command % (run_dir)
    command = ('%s %d --checkpoint_dir %s/checkpoints/%s '
               '--throughput_estimation_interval %d') % (parameterized_command,
                                                         job.num_steps,
                                                         checkpoint_dir,
                                                         job.checkpoint_dir,
                                                         interval)
    try:
        output = subprocess.run(command,
                                env=env,
                                #stdout=subprocess.PIPE,
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
                    start_steps = int(steps)
                elif float(time) > earliest_end_time:
                    break
        if start_time is None:
            return (-1, -1, -1)
        throughput = (int(steps) - start_steps) / (float(time) - start_time)
        if throughputs is None:
            throughputs = (throughput,)
        else:
            throughputs += (throughput,)

    return throughputs

def main(args):
    global job_table

    if args.enable_mps:
        enable_mps()

    # initialize throughput dictionary
    throughputs = {}
    for job in job_table:
        throughputs[job.model] = {}

    outfile = open(args.output_file, 'w')

    # Get isolated throughputs.
    if args.measure_isolated_throughputs:
        for i in range(len(job_table)):

          # Increase the number of steps until the runtime exceeds
          # the specified time per measurement.
          while True:
              print('%s] Running %s in isolation for '
                    '%d steps' % (datetime.datetime.now(),
                                  job_table[i].model,
                                  job_table[i].num_steps))
              start_time = time.time()
              output = run_job(job_table[i], args.run_dir, args.data_dir, args.checkpoint_dir)
              if not output:
                  break
              runtime = time.time() - start_time
              if runtime < args.seconds_per_measurement:
                  job_table[i].num_steps *= 2
              else:
                  break

          if not output:
              continue
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
          #throughputs[job_table[i].model] = {}
          throughputs[job_table[i].model][JobTypeSet((job_table[i].model,))] = throughput
          outfile.write("%s, %.3f\n"%(job_table[i].model, throughput))
          print('%s: %.3f (%.3f seconds)\n' % (job_table[i].model, throughput,
                                               runtime))
    return

    # uncomment to add a prompt if needed
    #input("Press Enter to continue...")

    # Get co-located throughputs.
    for i in range(len(job_table)):
        for j in range(i,len(job_table)):
            for k in range(j,len(job_table)):
                job1 = copy.deepcopy(job_table[i])
                job2 = copy.deepcopy(job_table[j])
                job3 = copy.deepcopy(job_table[k])

                job1.checkpoint_dir = '_1'
                job2.checkpoint_dir = '_2'
                job3.checkpoint_dir = '_3'

                if ((job1.model in throughputs and
                    job2.model in throughputs[job1.model]) or
                    (job2.model in throughputs and
                    job1.model in throughputs[job2.model])):
                    continue
                if job1.model not in throughputs:
                    throughputs[job1.model] = {}
                if job2.model not in throughputs:
                    throughputs[job2.model] = {}
                print('%s] Running %s with %s and %s' % (datetime.datetime.now(),
                                                  job1.model, job2.model, job3.model))
                outputs = []
                results = []
                pipe_list = []
                pool = multiprocessing.Pool(3)
                for job in [job1, job2, job3]:
                    results.append(pool.starmap_async(run_job,
                                                      [(job, args.run_dir, args.data_dir, args.checkpoint_dir),]))
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
                key = '(%s, %s, %s)' % (job1.model, job2.model, job3.model)
                if success:
                    throughput = get_throughputs(outputs)
                else:
                    throughput = (-1, -1, -1)

                throughputs_key = JobTypeSet((job1.model,job2.model,job3.model))
                throughputs[job1.model][throughputs_key] = throughput[0]
                throughputs[job2.model][throughputs_key] = throughput[1]
                throughputs[job3.model][throughputs_key] = throughput[2]
             
                outfile.write("%s, %s, %s, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n"%(job1.model, job2.model, job3.model,
                                                        throughput[0],
                                                        throughput[1],
                                                        throughput[2],
                                                        throughput[0]/throughputs[job1.model][JobTypeSet((job1.model,))],
                                                        throughput[1]/throughputs[job2.model][JobTypeSet((job2.model,))],
                                                        throughput[2]/throughputs[job3.model][JobTypeSet((job3.model,))]))

                print('(%s, %s, %s): (%.3f, %.3f, %.3f)\n' % (job1.model, job2.model, job3.model,
                                                    throughput[0],
                                                    throughput[1],
                                                    throughput[2]))
    outfile.close()
"""
    with open(args.output_file, 'w') as f:
        for key1 in throughputs.keys():
            for key2 in throughputs[key1].keys():
                if len(key2) == 3:
                    f.write("%s, %s, %s, %.3f, %.3f, %.3f\n"%(key2[0], key2[1], key2[2], 
                                                        throughputs[key2[0]][key2],
                                                        throughputs[key2[1]][key2],
                                                        throughputs[key2[2]][key2]))
"""
#        f.write(jsons.dumps(throughputs, indent=4))

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
    parser.add_argument('-r', '--run_dir', type=str,
                        default='/lfs/1/keshav2/workspace',
                        help='Directory to run from')
    parser.add_argument('-d', '--data_dir', type=str,
                        default=None,
                        help='Directory where data is stored')
    parser.add_argument('-c', '--checkpoint_dir', type=str,
                        default=None,
                        help="Directory for checkpoints")
    args = parser.parse_args()
    main(args)
