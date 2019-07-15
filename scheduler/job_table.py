from job_template import JobTemplate

JobTable = [
    JobTemplate(model='ResNet-18',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/cifar10 && python3 '
                       'main.py --data_dir=%s/data/cifar10'),
              num_steps_arg='--num_steps'),
    JobTemplate(model='ResNet-50',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/imagenet && python3 '
                       'main.py -j 4 -a resnet50 -b 64 '
                       '%s/data/imagenet/pytorch'),
              num_steps_arg='--num_minibatches'),
    JobTemplate(model='Transformer',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'translation && python3 train.py -data '
                       '%s/data/translation/multi30k.atok.low.pt'
                       '-proj_share_weight'),
              num_steps_arg='-step'),
    JobTemplate(model='A3C',
              command=('cd %s/gpusched/workloads/pytorch/rl && '
                       'python3 main.py --env PongDeterministic-v4 --workers 4 '
                       '--amsgrad True'),
              num_steps_arg='--max-steps',
              needs_data_dir=False),
    JobTemplate(model='Recommendation',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'recommendation/scripts/ml-20m && python3 train.py '
                       '--data_dir %s/data/ml-20m/pro_sg/'),
              num_steps_arg='-n'),
    JobTemplate(model='CycleGAN',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'cyclegan && python3 cyclegan.py --dataset_path '
                       '%s/data/monet2photo --decay_epoch 0'),
              num_steps_arg='--n_steps'),
]
