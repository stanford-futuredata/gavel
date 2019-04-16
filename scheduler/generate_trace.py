import argparse
import numpy as np
import random

jobs = [
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
      'image_classification/imagenet && python3 '
      'main.py -j 16 -a resnet50 -b 64 /home/deepakn94/imagenet/'),
     '--num_minibatches', 1000),
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/'
      'image_classification/cifar10 && python3 '
      'main.py --data_dir=/home/keshavsanthanam/data/cifar10 '),
      '--num_epochs', 500),
    (('cd /home/keshavsanthanam/gpusched/workloads/pytorch/rl && '
       'python3 main.py --env PongDeterministic-v4 --workers 8 --amsgrad True',
      '--max-steps', 1000))
]

def generate(lam, N, output_file):
    samples = np.random.poisson(lam, size=N)
    arrival_times = [np.sum(samples[0:i]) for i in xrange(N)] 
    with open(output_file, 'w') as f:
        for arrival_time in arrival_times:
            job = random.choice(jobs)
            f.write('%s\t%s\t%d\t%d\n' % (job[0], job[1], job[2], arrival_time))

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

