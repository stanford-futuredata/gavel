import argparse
import numpy as np
import random

INITIAL_DELAY = 10
MAX_JOB_STEPS = 1000000

jobs = [
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
      'image_classification/imagenet && python3 '
      'main.py -j 4 -a resnet50 -b 64 /home/deepakn94/imagenet/'),
     '--num_minibatches'),
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/rl && '
       'python3 main.py --env PongDeterministic-v4 --workers 8 --amsgrad True',
      '--max-steps')),
    (('cd /home/keshavsanthanam/gpusched/workloads/tensorflow/'
      'language_modeling && python ptb_word_lm.py '
      '--data_path=/home/keshavsanthanam/data/ptb --model=medium',
      '--max_steps')),
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
      'recommendation/scripts/ml-20m && python3 train.py'), '-n'),
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
      'image_classification/cifar10 && python3 '
      'main.py --data_dir=/home/keshavsanthanam/data/cifar10 '),
      '--num_epochs'),
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/translation/ '
      '&& python3 train.py -data /home/keshavsanthanam/data/translation/'
      'multi30k.atok.low.pt -proj_share_weight'), '-epoch'),
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/cyclegan '
      '&& python3 cyclegan.py --dataset_path /home/keshavsanthanam/data'
      '/monet2photo --decay_epoch 0'), '--n_steps'),
]
"""
    (('cd /home/keshavsanthanam/gpusched/workloads/tensorflow/translation/ '
      '&& python -m nmt.nmt --src=vi --tgt=en '
      '--vocab_prefix=/home/keshavsanthanam/data/nmt_data/vocab '
      '--train_prefix=/home/keshavsanthanam/data/nmt_data/train '
      '--dev_prefix=/home/keshavsanthanam/data/nmt_data/tst2012 '
      '--test_prefix=/home/keshavsanthanam/data/nmt_data/tst2013 '
      '--num_layers=2 --num_units=128 --dropout=0.2 '
      '--metrics=bleu'), '--num_train_steps'),
"""

def generate(lam, N, output_file):
    samples = np.random.poisson(lam, size=N)
    arrival_times = [np.sum(samples[0:i]) for i in xrange(N)]
    with open(output_file, 'w') as f:
        for arrival_time in arrival_times:
            job = random.choice(jobs)
            duration = 10 ** random.uniform(0, 3)  # this is in minutes.
            duration *= 60
            f.write('%s\t%s\t%d\t%d\t%d\n' % (job[0], job[1], MAX_JOB_STEPS,
                    arrival_time + INITIAL_DELAY, duration))

def main(args):
    generate(args.lam, args.num_jobs, args.output_file)

if __name__=='__main__':
   parser = argparse.ArgumentParser(description='Generate scheduler trace')
   parser.add_argument('-l', '--lam', type=float, default=1,
                       help='Lambda value for Poisson arrival process')
   parser.add_argument('-n', '--num_jobs', type=int, default=50,
                       help='Number of jobs')
   parser.add_argument('-o', '--output_file', type=str, required=True,
                       help='File to output trace to')
   main(parser.parse_args())
