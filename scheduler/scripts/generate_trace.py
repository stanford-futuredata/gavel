import argparse
import numpy as np
import random

INITIAL_DELAY = 10
MAX_JOB_STEPS = 10000000

random.seed(42)
np.random.seed(42)

jobs = [
    ('ResNet-50', ('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
      'image_classification/imagenet && python3 '
      'main.py -j 4 -a resnet50 -b 64 /home/deepakn94/imagenet/'),
     '--num_minibatches'),
    ('A3C', ('cd /home/keshavsanthanam/gpusched/workloads/pytorch/rl && '
       'python3 main.py --env PongDeterministic-v4 --workers 4 --amsgrad True'),
      '--max-steps'),
    ('LM', ('cd /home/keshavsanthanam/gpusched/workloads/pytorch/language_modeling '
      '&& python main.py --cuda --data /home/keshavsanthanam/data/wikitext-2'),
      '--steps'),
    ('Recommendation', ('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
      'recommendation/scripts/ml-20m && python3 train.py'), '-n'),
    ('ResNet-18', ('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
      'image_classification/cifar10 && python3 '
      'main.py --data_dir=/home/keshavsanthanam/data/cifar10'),
      '--num_steps'),
    ('Transformer', ('cd /home/keshavsanthanam/gpusched/workloads/pytorch/translation/ '
      '&& python3 train.py -data /home/keshavsanthanam/data/translation/'
      'multi30k.atok.low.pt -proj_share_weight'), '-step'),
    ('CycleGAN', ('cd /home/keshavsanthanam/gpusched/workloads/pytorch/cyclegan '
      '&& python3 cyclegan.py --dataset_path /home/keshavsanthanam/data'
      '/monet2photo --decay_epoch 0'), '--n_steps'),
]

throughputs = {
        'ResNet-50': 1.333386669,
        'ResNet-18': 46.45544922,
        'CycleGAN': 3.497604141,
        'A3C': 6.035731531,
        'Transformer': 8.26514588,
        'Recommendation': 0.4535558781,
        'LM': 0.7638835841,
}

"""
    (('cd /home/keshavsanthanam/gpusched/workloads/tensorflow/translation/ '
      '&& python -m nmt.nmt --src=vi --tgt=en '
      '--vocab_prefix=/home/keshavsanthanam/data/nmt_data/vocab '
      '--train_prefix=/home/keshavsanthanam/data/nmt_data/train '
      '--dev_prefix=/home/keshavsanthanam/data/nmt_data/tst2012 '
      '--test_prefix=/home/keshavsanthanam/data/nmt_data/tst2013 '
      '--num_layers=2 --num_units=128 --dropout=0.2 '
      '--metrics=bleu'), '--num_train_steps'),
    (('cd /home/keshavsanthanam/gpusched/workloads/tensorflow/'
      'language_modeling && python ptb_word_lm.py '
      '--data_path=/home/keshavsanthanam/data/ptb --model=medium',
      '--max_steps')),
"""

def generate(lam, N, initial_delay, output_file):
    samples = np.random.poisson(lam, size=N)
    arrival_times = [np.sum(samples[0:i]) for i in range(N)]
    with open(output_file, 'w') as f:
        for arrival_time in arrival_times:
            job = random.choice(jobs)
            duration = 10 ** random.uniform(1.5, 4)  # this is in minutes.
            duration *= 60
            total_steps = int(duration * throughputs[job[0]])
            scale_factor = 1
            # 'job_type\tcommand\tnum_steps_arg\ttotal_steps\tarrival_time'
            f.write('%s\t%s\t%s\t%d\t%d\t%d\n' % (job[0], job[1], job[2],
                                                  total_steps,
                                                  arrival_time + initial_delay,
                                                  scale_factor))

def main(args):
    generate(args.lam, args.num_jobs, args.initial_delay, args.output_file)

if __name__=='__main__':
   parser = argparse.ArgumentParser(description='Generate scheduler trace')
   parser.add_argument('-l', '--lam', type=float, default=1,
                       help='Lambda value for Poisson arrival process')
   parser.add_argument('-n', '--num_jobs', type=int, default=50,
                       help='Number of jobs')
   parser.add_argument('-o', '--output_file', type=str, required=True,
                       help='File to output trace to')
   parser.add_argument('-i', '--initial_delay', type=int, default=10,
                       help='Initial job arrival time delay')
   main(parser.parse_args())
