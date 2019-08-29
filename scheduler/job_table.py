from job_template import JobTemplate

# TODO: Figure out a better way of toggling between the two tables.
THROUGHPUT_ESTIMATION = True

if THROUGHPUT_ESTIMATION:
    JobTable = [
        JobTemplate(model='ResNet-18 (batch size 16)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/cifar10 && python3 '
                           'main.py --data_dir=%s/data/cifar10'),
                  num_steps_arg='--num_steps'),
        JobTemplate(model='ResNet-18 (batch size 32)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/cifar10 && python3 '
                           'main.py --data_dir=%s/data/cifar10'),
                  num_steps_arg='--num_steps'),
        JobTemplate(model='ResNet-18 (batch size 64)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/cifar10 && python3 '
                           'main.py --data_dir=%s/data/cifar10'),
                  num_steps_arg='--num_steps'),
        JobTemplate(model='ResNet-18 (batch size 128)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/cifar10 && python3 '
                           'main.py --data_dir=%s/data/cifar10'),
                  num_steps_arg='--num_steps'),
        JobTemplate(model='ResNet-18 (batch size 256)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/cifar10 && python3 '
                           'main.py --data_dir=%s/data/cifar10'),
                  num_steps_arg='--num_steps'),
        JobTemplate(model='ResNet-50 (batch size 16)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/imagenet && python3 '
                           'main.py -j 4 -a resnet50 -b 64 '
                           '%s/data/imagenet/pytorch'),
                  num_steps_arg='--num_minibatches'),
        JobTemplate(model='ResNet-50 (batch size 32)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/imagenet && python3 '
                           'main.py -j 4 -a resnet50 -b 64 '
                           '%s/data/imagenet/pytorch'),
                  num_steps_arg='--num_minibatches'),
        JobTemplate(model='ResNet-50 (batch size 64)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/imagenet && python3 '
                           'main.py -j 4 -a resnet50 -b 64 '
                           '%s/data/imagenet/pytorch'),
                  num_steps_arg='--num_minibatches'),
        JobTemplate(model='ResNet-50 (batch size 128)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'image_classification/imagenet && python3 '
                           'main.py -j 4 -a resnet50 -b 64 '
                           '%s/data/imagenet/pytorch'),
                  num_steps_arg='--num_minibatches'),
        JobTemplate(model='Transformer (batch size 16)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'translation && python3 train.py -data '
                           '%s/data/translation/multi30k.atok.low.pt'
                           '-proj_share_weight'),
                  num_steps_arg='-step'),
        JobTemplate(model='Transformer (batch size 32)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'translation && python3 train.py -data '
                           '%s/data/translation/multi30k.atok.low.pt'
                           '-proj_share_weight'),
                  num_steps_arg='-step'),
        JobTemplate(model='Transformer (batch size 64)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'translation && python3 train.py -data '
                           '%s/data/translation/multi30k.atok.low.pt'
                           '-proj_share_weight'),
                  num_steps_arg='-step'),
        JobTemplate(model='Transformer (batch size 128)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'translation && python3 train.py -data '
                           '%s/data/translation/multi30k.atok.low.pt'
                           '-proj_share_weight'),
                  num_steps_arg='-step'),
        JobTemplate(model='Transformer (batch size 256)',
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
        JobTemplate(model='LM (batch size 5)',
                    command=('cd %s && ./placeholder_command'),
                    num_steps_arg='--placeholder_steps',
                    needs_data_dir=False),
        JobTemplate(model='LM (batch size 10)',
                    command=('cd %s && ./placeholder_command'),
                    num_steps_arg='--placeholder_steps',
                    needs_data_dir=False),
        JobTemplate(model='LM (batch size 20)',
                    command=('cd %s && ./placeholder_command'),
                    num_steps_arg='--placeholder_steps',
                    needs_data_dir=False),
        JobTemplate(model='LM (batch size 40)',
                    command=('cd %s && ./placeholder_command'),
                    num_steps_arg='--placeholder_steps',
                    needs_data_dir=False),
        JobTemplate(model='LM (batch size 80)',
                    command=('cd %s && ./placeholder_command'),
                    num_steps_arg='--placeholder_steps',
                    needs_data_dir=False),
        JobTemplate(model='Recommendation (batch size 512)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'recommendation/scripts/ml-20m && python3 train.py '
                           '--data_dir %s/data/ml-20m/pro_sg/'),
                  num_steps_arg='-n'),
        JobTemplate(model='Recommendation (batch size 1024)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'recommendation/scripts/ml-20m && python3 train.py '
                           '--data_dir %s/data/ml-20m/pro_sg/'),
                  num_steps_arg='-n'),
        JobTemplate(model='Recommendation (batch size 2048)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'recommendation/scripts/ml-20m && python3 train.py '
                           '--data_dir %s/data/ml-20m/pro_sg/'),
                  num_steps_arg='-n'),
        JobTemplate(model='Recommendation (batch size 4096)',
                  command=('cd %s/gpusched/workloads/pytorch/'
                           'recommendation/scripts/ml-20m && python3 train.py '
                           '--data_dir %s/data/ml-20m/pro_sg/'),
                  num_steps_arg='-n'),
        JobTemplate(model='Recommendation (batch size 8192)',
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
else:
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
