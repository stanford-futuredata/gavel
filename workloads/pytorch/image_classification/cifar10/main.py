'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import math
import sys
import time

from models import *
from utils import progress_bar

# TODO: Figure out a cleaner way of including gavel_iterator.
imagenet_dir = os.path.dirname(os.path.realpath(__file__))
image_classification_dir = os.path.dirname(imagenet_dir)
pytorch_dir = os.path.dirname(image_classification_dir)
workloads_dir = os.path.dirname(pytorch_dir)
gpusched_dir = os.path.dirname(workloads_dir)
scheduler_dir = os.path.join(gpusched_dir, 'scheduler')
sys.path.append(scheduler_dir)
from gavel_iterator import GavelIterator

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data_dir', required=True, type=str, help='Data directory')
parser.add_argument('--num_epochs', default=None, type=int, help='Number of epochs to train for')
parser.add_argument('--num_steps', default=None, type=int, help='Number of steps to train for')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--checkpoint_dir', default='/lfs/1/keshav2/checkpoints/resnet-18',
                    type=str, help='Checkpoint directory')
parser.add_argument('--use_progress_bar', '-p', action='store_true', default=False, help='Use progress bar')
parser.add_argument('--log_interval', type=int, default=100,
                    help='Interval to log')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='Distributed backend')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Local rank')
parser.add_argument('--rank', default=None, type=int,
                    help='Rank')
parser.add_argument('--world_size', default=None, type=int,
                    help='World size')
parser.add_argument('--master_addr', default=None, type=str,
                    help='Master address to use for distributed run')
parser.add_argument('--master_port', default=None, type=int,
                    help='Master port to use for distributed run')
parser.add_argument('--throughput_estimation_interval', type=int, default=None,
                    help='Steps between logging steps completed')
parser.add_argument('--max_duration', type=int, default=None,
                    help='Maximum duration in seconds')
parser.add_argument('--enable_gavel_iterator', action='store_true',
                    default=False, help='If set, use Gavel iterator')

args = parser.parse_args()


print('==> Starting script..')

torch.cuda.set_device(args.local_rank)
if args.num_epochs is not None and args.num_steps is not None:
    raise ValueError('Only one of num_epochs and num_steps may be set')
elif args.num_epochs is None and args.num_steps is None:
    raise ValueError('One of num_epochs and num_steps must be set')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')
net = ResNet18()
net = net.cuda()

args.distributed = False
if args.master_addr is not None:
    args.distributed = True
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank],
            output_device=args.local_rank)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_train)
train_sampler = None
if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=2,
                                          sampler=train_sampler)

testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


cumulative_steps = 0
cumulative_time = 0

def load_checkpoint(args, checkpoint_path):
    # Load checkpoint.
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path,
                                    map_location='cuda:{}'.format(args.local_rank))
            print('==> Resuming from checkpoint at %s...' % (checkpoint_path))
            return checkpoint
        except Exception as e:
            print('Error reading checkpoint: %s' % (e))
            return None
    return None

def save_checkpoint(checkpoint_path, state):
    print('==> Saving checkpoint at %s...' % (checkpoint_path))
    torch.save(state, checkpoint_path)

if args.checkpoint_dir is not None:
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.chkpt')

if args.enable_gavel_iterator:
    trainloader = GavelIterator(trainloader, args.checkpoint_dir,
                                load_checkpoint_func=load_checkpoint,
                                save_checkpoint_func=save_checkpoint)
    checkpoint = trainloader.load_checkpoint(args, checkpoint_path)
else:
    checkpoint = load_checkpoint(args, checkpoint_path)
if checkpoint is not None:
    net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch, cumulative_steps=None, cumulative_time=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    done = False
    finished_epoch = True
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.use_progress_bar:
          progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        elif batch_idx % args.log_interval == 0 and batch_idx > 0:
            print('Batch: %d, Loss: %.3f, '
                  'Acc: %.3f%% (%d/%d)' % (batch_idx, train_loss/(batch_idx+1),
                                           100.*correct/total, correct, total))
        if cumulative_time is not None:
            cumulative_time += time.time() - start_time
            if (args.max_duration is not None and
                cumulative_time > args.max_duration):
                done = True
                finished_epoch = False
            start_time = time.time()
        if cumulative_steps is not None:
            cumulative_steps += 1
            if (args.throughput_estimation_interval is not None and
                cumulative_steps % args.throughput_estimation_interval == 0):
                print('[THROUGHPUT_ESTIMATION]\t%s\t%d' % (time.time(),
                                                           cumulative_steps))
            if args.num_steps is not None and cumulative_steps >= args.num_steps:
                done = True
                finished_epoch = False
                break
    return (cumulative_steps, cumulative_time, done, finished_epoch)



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving checkpoint at %s...' % (checkpoint_path))
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        torch.save(state, checkpoint_path)
        best_acc = acc

if args.num_steps is not None:
    args.num_epochs = math.ceil(float(args.num_steps) *
                                args.batch_size / len(trainloader))

for epoch in range(start_epoch, args.num_epochs):
    (cumulative_steps, cumulative_time, done, finished_epoch) =\
            train(epoch, cumulative_steps, cumulative_time)
    if args.enable_gavel_iterator:
        if trainloader.done:
            break
        elif done:
            trainloader.complete()
            break
    elif done:
        break

state = {
    'net': net.state_dict(),
    'epoch': epoch,
}
if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
if not args.distributed or args.rank == 0:
    if args.enable_gavel_iterator:
        trainloader.save_checkpoint(checkpoint_path, state)
    else:
        save_checkpoint(checkpoint_path, state)
