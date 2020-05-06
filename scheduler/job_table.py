from job_template import JobTemplate

JobTable = [
    JobTemplate(model='ResNet-18 (batch size 16)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/cifar10 && python3 '
                       'main.py --data_dir=%s/data/cifar10 --batch_size 16 '),
              num_steps_arg='--num_steps', distributed=True),
    JobTemplate(model='ResNet-18 (batch size 32)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/cifar10 && python3 '
                       'main.py --data_dir=%s/data/cifar10 --batch_size 32 '),
              num_steps_arg='--num_steps', distributed=True),
    JobTemplate(model='ResNet-18 (batch size 64)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/cifar10 && python3 '
                       'main.py --data_dir=%s/data/cifar10 --batch_size 64 '),
              num_steps_arg='--num_steps', distributed=True),
    JobTemplate(model='ResNet-18 (batch size 128)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/cifar10 && python3 '
                       'main.py --data_dir=%s/data/cifar10 --batch_size 128 '),
              num_steps_arg='--num_steps', distributed=True),
    JobTemplate(model='ResNet-18 (batch size 256)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/cifar10 && python3 '
                       'main.py --data_dir=%s/data/cifar10 --batch_size 256 '),
              num_steps_arg='--num_steps', distributed=True),
    JobTemplate(model='ResNet-50 (batch size 16)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/imagenet && python3 '
                       'main.py -j 8 -a resnet50 -b 16 '
                       '%s/data/imagenet/pytorch'),
              num_steps_arg='--num_minibatches', distributed=True),
    JobTemplate(model='ResNet-50 (batch size 32)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/imagenet && python3 '
                       'main.py -j 8 -a resnet50 -b 32 '
                       '%s/data/imagenet/pytorch'),
              num_steps_arg='--num_minibatches', distributed=True),
    JobTemplate(model='ResNet-50 (batch size 64)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/imagenet && python3 '
                       'main.py -j 8 -a resnet50 -b 64 '
                       '%s/data/imagenet/pytorch'),
              num_steps_arg='--num_minibatches', distributed=True),
    JobTemplate(model='ResNet-50 (batch size 128)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'image_classification/imagenet && python3 '
                       'main.py -j 8 -a resnet50 -b 128 '
                       '%s/data/imagenet/pytorch'),
              num_steps_arg='--num_minibatches', distributed=True),
    JobTemplate(model='Transformer (batch size 16)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'translation && python3 train.py -data '
                       '%s/data/translation/multi30k.atok.low.pt '
                       '-batch_size 16 -proj_share_weight'),
              num_steps_arg='-step', distributed=True),
    JobTemplate(model='Transformer (batch size 32)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'translation && python3 train.py -data '
                       '%s/data/translation/multi30k.atok.low.pt '
                       '-batch_size 32 -proj_share_weight'),
              num_steps_arg='-step', distributed=True),
    JobTemplate(model='Transformer (batch size 64)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'translation && python3 train.py -data '
                       '%s/data/translation/multi30k.atok.low.pt '
                       '-batch_size 64 -proj_share_weight'),
              num_steps_arg='-step', distributed=True),
    JobTemplate(model='Transformer (batch size 128)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'translation && python3 train.py -data '
                       '%s/data/translation/multi30k.atok.low.pt '
                       '-batch_size 128 -proj_share_weight'),
              num_steps_arg='-step', distributed=True),
    JobTemplate(model='Transformer (batch size 256)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'translation && python3 train.py -data '
                       '%s/data/translation/multi30k.atok.low.pt '
                       '-batch_size 256 -proj_share_weight'),
              num_steps_arg='-step', distributed=True),
    JobTemplate(model='A3C',
              command=('cd %s/gpusched/workloads/pytorch/rl && '
                       'python3 main.py --env PongDeterministic-v4 --workers 4 '
                       '--amsgrad True'),
              num_steps_arg='--max-steps',
              needs_data_dir=False),
    JobTemplate(model='LM (batch size 5)',
                command=('cd %s/gpusched/workloads/pytorch/'
                         'language_modeling && python main.py --cuda --data '
                         '%s/data/wikitext2 --batch_size 5'),
                num_steps_arg='--steps', distributed=True),
    JobTemplate(model='LM (batch size 10)',
                command=('cd %s/gpusched/workloads/pytorch/'
                         'language_modeling && python main.py --cuda --data '
                         '%s/data/wikitext2 --batch_size 10'),
                num_steps_arg='--steps', distributed=True),
    JobTemplate(model='LM (batch size 20)',
                command=('cd %s/gpusched/workloads/pytorch/'
                         'language_modeling && python main.py --cuda --data '
                         '%s/data/wikitext2 --batch_size 20'),
                num_steps_arg='--steps', distributed=True),
    JobTemplate(model='LM (batch size 40)',
                command=('cd %s/gpusched/workloads/pytorch/'
                         'language_modeling && python main.py --cuda --data '
                         '%s/data/wikitext2 --batch_size 40'),
                num_steps_arg='--steps', distributed=True),
    JobTemplate(model='LM (batch size 80)',
                command=('cd %s/gpusched/workloads/pytorch/'
                         'language_modeling && python main.py --cuda --data '
                         '%s/data/wikitext2 --batch_size 80'),
                num_steps_arg='--steps', distributed=True),
    JobTemplate(model='Recommendation (batch size 512)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'recommendation/ && python3 train.py '
                       '--data_dir %s/data/ml-20m/pro_sg/ --batch_size 512'),
              num_steps_arg='-n'),
    JobTemplate(model='Recommendation (batch size 1024)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'recommendation/ && python3 train.py '
                       '--data_dir %s/data/ml-20m/pro_sg/ --batch_size 1024'),
              num_steps_arg='-n'),
    JobTemplate(model='Recommendation (batch size 2048)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'recommendation/ && python3 train.py '
                       '--data_dir %s/data/ml-20m/pro_sg/ --batch_size 2048'),
              num_steps_arg='-n'),
    JobTemplate(model='Recommendation (batch size 4096)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'recommendation/ && python3 train.py '
                       '--data_dir %s/data/ml-20m/pro_sg/ --batch_size 4096'),
              num_steps_arg='-n'),
    JobTemplate(model='Recommendation (batch size 8192)',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'recommendation/ && python3 train.py '
                       '--data_dir %s/data/ml-20m/pro_sg/ --batch_size 8192'),
              num_steps_arg='-n'),
    JobTemplate(model='CycleGAN',
              command=('cd %s/gpusched/workloads/pytorch/'
                       'cyclegan && python3 cyclegan.py --dataset_path '
                       '%s/data/monet2photo --decay_epoch 0'),
              num_steps_arg='--n_steps'),
]
