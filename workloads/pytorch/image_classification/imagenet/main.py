import argparse
import os
import random
import shutil
import time
import warnings
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

###
import torch_xla
import torch_xla.core.xla_model as xm
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'

class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        print(self.target)
        self.length = length
    def __getitem__(self, index):
        return self.tensor, self.target
    def __len__(self):
        return self.length


###

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_minibatches', default=None, type=int,
                    help='number of minibatches to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                   metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint_dir',
                    type=str,
                    default='/lfs/1/keshav2/checkpoints/resnet-50',
                    help='Directory for checkpoints')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--throughput_estimation_interval', type=int, default=None,
                    help='Steps between logging steps completed')
parser.add_argument('--timeout', type=int, default=None,
                    help='Timeout (in seconds)')
parser.add_argument('--synth', dest='synthetic_data', action='store_true', default=False,
                    help='synthetic data flag')

best_acc1 = 0
total_minibatches = 0
cumulative_seconds = 0

def main():
    global args, best_acc1, total_minibatches, cumulative_seconds
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = False
    if args.master_addr is not None:
        args.distributed = True
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.master_port)
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

###
#    torch.cuda.set_device(args.local_rank)
#    model = model.cuda()
     dev = xm.xla_device()
     model = model.to(dev)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # define loss function (criterion) and optimizer
#    criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
###

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Load from checkpoint.
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.chkpt')
    if os.path.exists(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.synthetic_data:
        train_dataset = SyntheticDataset((3, 299, 299),1281167)
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
                                             train_dataset, 
                                             batch_size=args.batch_size,       
                                             shuffle=(train_sampler is None),
                                             num_workers=args.workers, 
                                             pin_memory=True, sampler=train_sampler)
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

###
#    x = np.random.randn(number_of_samples, sample_height, sample_width, 3).astype(np.float32)
#    y = np.eye(num_classes)[np.random.randint(num_classes, size=number_of_samples)].astype(np.float32)
###
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

###
    if args.synthetic_data:
        train_loader = torch.utils.data.DataLoader(
                                             train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=(train_sampler is None),
                                             num_workers=args.workers,
                                             pin_memory=True, sampler=train_sampler)
        val_loader = copy.deepcopy(train_loader)
    else: 
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
###

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        num_minibatches, cumulative_seconds, finished_epoch = \
            train(train_loader, model, criterion, optimizer,
                  epoch, total_minibatches,
                  max_minibatches=args.num_minibatches,
                  cumulative_seconds=cumulative_seconds,
                  timeout=args.timeout)
        total_minibatches += num_minibatches

        if (args.num_minibatches is not None and
            total_minibatches >= args.num_minibatches):
            break
        elif args.timeout is not None and cumulative_seconds >= args.timeout:
          break

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion)

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, checkpoint_path)


def train(train_loader, model, criterion, optimizer, epoch,
          total_minibatches, max_minibatches, cumulative_seconds,
          timeout):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    finished_epoch = True
    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        if (total_minibatches is not None and
            i + total_minibatches >= max_minibatches):
            finished_epoch = False
            break
        elif (timeout is not None and cumulative_seconds >= timeout):
          break
        cumulative_seconds += time.time() - start_time
        start_time = time.time()
        # measure data loading time
        data_time.update(time.time() - end)


###
        #input = input.cuda(non_blocking=True)
        #target = target.cuda(non_blocking=True)
        input = input.to(dev)
        target = target.to(dev)
###
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

        if (args.throughput_estimation_interval is not None and
            i % args.throughput_estimation_interval == 0):
            print('[THROUGHPUT_ESTIMATION]\t%s\t%d' % (time.time(),
                                                       total_minibatches+i+1))
    return i, cumulative_seconds, finished_epoch

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
###
            #input = input.cuda(non_blocking=True)
            #target = target.cuda(non_blocking=True)
            input = input.to(dev)
            target = target.to(dev)
###
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, checkpoint_filename):
    if checkpoint_filename is not None:
        torch.save(state, checkpoint_filename)
        print("=> saved checkpoint '{}'".format(checkpoint_filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
