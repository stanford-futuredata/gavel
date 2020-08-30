import os
import sys

from job_template import JobTemplate

def resnet18(batch_size, gavel_dir):
    model = 'ResNet-18 (batch size %d)' % (batch_size)
    command = 'python3 main.py --data_dir=%s/data/cifar10'
    command += ' --batch_size %d' % (batch_size)
    working_directory = \
        '%s/workloads/pytorch/image_classification/cifar10' % (gavel_dir)
    num_steps_arg = '--num_steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def resnet50(batch_size, gavel_dir):
    model = 'ResNet-50 (batch size %d)' % (batch_size)
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % (batch_size)
    command += ' %s/data/imagenet/pytorch'
    working_directory = \
        '%s/workloads/pytorch/image_classification/imagenet' % (gavel_dir)
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def transformer(batch_size, gavel_dir):
    model = 'Transformer (batch size %d)' % (batch_size)
    command = 'python3 train.py -data %s/data/translation/multi30k.atok.low.pt'
    command += '-batch_size %d -proj_share_weight' % (batch_size)
    working_directory = '%s/workloads/pytorch/translation' % (gavel_dir)
    num_steps_arg = '-step'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def lm(batch_size, gavel_dir):
    model = 'LM (batch size %d)' % (batch_size)
    command = 'python main.py --cuda --data %s/data/wikitext2'
    command += ' --batch_size %d' % (batch_size)
    working_directory = '%s/workloads/pytorch/language_modeling' % (gavel_dir)
    num_steps_arg = '--steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def recommendation(batch_size, gavel_dir):
    model = 'Recommendation (batch size %d)' % (batch_size)
    command = 'python3 train.py --data_dir %s/data/ml-20m/pro_sg/'
    command += ' --batch_size %d' % (batch_size)
    working_directory = '%s/workloads/pytorch/recommendation' % (gavel_dir)
    num_steps_arg = '-n'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg)

def a3c(gavel_dir):
    model = 'A3C'
    command = ('python3 main.py --env PongDeterministic-v4 --workers 4 '
               '--amsgrad True')
    working_directory = '%s/workloads/pytorch/rl' % (gavel_dir)
    num_steps_arg = '--max_steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg,
                       needs_data_dir=False)

def cyclegan(gavel_dir):
    model = 'CycleGAN'
    working_directory = '%s/workloads/pytorch/cyclegan' % (gavel_dir)
    command = ('python3 cyclegan.py --dataset_path %s/data/monet2photo'
               ' --decay_epoch 0')
    num_steps_arg = '--n_steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg)

JobTable = []

file_path = os.path.realpath(__file__)
gavel_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

for batch_size in [16, 32, 64, 128, 256]:
    JobTable.append(resnet18(batch_size, gavel_dir))
for batch_size in [16, 32, 64, 128]:
    JobTable.append(resnet50(batch_size, gavel_dir))
for batch_size in [16, 32, 64, 128, 256]:
    JobTable.append(transformer(batch_size, gavel_dir))
for batch_size in [5, 10, 20, 40, 80]:
    JobTable.append(lm(batch_size, gavel_dir))
for batch_size in [512, 1024, 2048, 4096, 8192]:
    JobTable.append(recommendation(batch_size, gavel_dir))
JobTable.append(a3c(gavel_dir))
JobTable.append(cyclegan(gavel_dir))
