# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.onnx
import sys
import multiprocessing as mp

import data
import model

# TODO: Figure out a cleaner way of including gavel_iterator.
lm_dir = os.path.dirname(os.path.realpath(__file__))
pytorch_dir = os.path.dirname(lm_dir)
workloads_dir = os.path.dirname(pytorch_dir)
gpusched_dir = os.path.dirname(workloads_dir)
scheduler_dir = os.path.join(gpusched_dir, 'scheduler')
sys.path.append(scheduler_dir)
from gavel_iterator import GavelIterator

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=None,
                    help='upper epoch limit')
parser.add_argument('--steps', type=int, default=None,
                    help='upper steps limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--checkpoint_dir', type=str,
                    default=None,
                    help='Checkpoint dir')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--throughput_estimation_interval', type=int, default=None,
                    help='Steps between logging steps completed')

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
parser.add_argument('--max_duration', type=int, default=None,
                    help='Maximum duration in seconds')
parser.add_argument('--job_id', type=int, default=None, help='Job ID')
parser.add_argument('--worker_id', type=int, default=None, help='Worker ID')
parser.add_argument('--sched_addr', type=str, default=None,
                    help='Scheduler server')
parser.add_argument('--sched_port', type=int, default=None,
                    help='Scheduler port')

args = parser.parse_args()

if args.epochs is not None and args.steps is not None:
    raise ValueError('Only one of epochs and steps may be set')
elif args.epochs is None and args.steps is None:
    raise ValueError('One of epochs and steps must be set')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

torch.cuda.set_device(args.local_rank)
device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

class CorpusDataset(torch.utils.data.Dataset):
    def __init__(self, data, batch_size, bptt):
        self._data = data.narrow(0, 0, (data.size(0) // batch_size) * batch_size)
        # Evenly divide the data across the bsz batches.
        self._data = self._data.view(batch_size, -1).t().contiguous().to(device)
        self._data_length = data.size(0)
        self._batch_size = batch_size
        self._bptt = bptt

    # get_input subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.
    def get_input(self, row_idx, col_idx):
        row_idx = row_idx % len(self._data)
        seq_len = min(self._bptt, len(self._data) - 1 - row_idx)
        data = self._data[row_idx: row_idx+seq_len, col_idx]
        target = self._data[row_idx+1: row_idx+1+seq_len, col_idx].view(data.size())
        data = torch.cat([data, data.new_zeros(self._bptt - data.size(0))])
        target = torch.cat([target, target.new_zeros(self._bptt - target.size(0))])
        return data, target

    def __len__(self):
        return self._data_length // self._bptt

    def __getitem__(self, idx):
        return self.get_input((idx // self._batch_size) * self._bptt,
                              idx % self._batch_size)

eval_batch_size = 10
train_dataset = CorpusDataset(corpus.train,
                              args.batch_size,
                              args.bptt)
val_dataset = CorpusDataset(corpus.valid,
                            eval_batch_size,
                            args.bptt)
test_dataset = CorpusDataset(corpus.test,
                             eval_batch_size,
                             args.bptt)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

load_from_checkpoint = False
if args.checkpoint_dir is not None:
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'model.chkpt')
        if os.path.exists(checkpoint_path):
            print('Loading checkpoint from %s...' % (checkpoint_path))
            with open(checkpoint_path, 'rb') as f:
                state = torch.load(f)
                model = state['model'].to(device)
            load_from_checkpoint = True
if not load_from_checkpoint:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                           args.nlayers, args.dropout, args.tied).to(device)

args.distributed = False
if args.master_addr is not None:
    args.distributed = True
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                output_device=args.local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=False)
else:
    train_sampler = None

args.enable_gavel_iterator = False
if args.job_id is not None:
    args.enable_gavel_iterator = True

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           sampler=train_sampler,
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=eval_batch_size,
                                         shuffle=False,
                                         drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=eval_batch_size,
                                          shuffle=False,
                                          drop_last=True)

if args.enable_gavel_iterator:
    train_loader = GavelIterator(train_loader, args.job_id, args.worker_id,
                                 args.distributed,
                                 args.sched_addr, args.sched_port)


cumulative_steps = 0
cumulative_time = 0

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), args.lr)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(loader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            (data, targets) = batch
            data = data.t()
            targets = targets.t()

            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets.flatten()).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(loader) - 1)


def train(cumulative_steps=None, cumulative_time=None):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.distributed:
        hidden = model.module.init_hidden(args.batch_size)
    else:
        hidden = model.init_hidden(args.batch_size)
    done = False
    for i, batch in enumerate(train_loader):
        total_duration_tracker_start = time.time()

        # Batch size should be the second dimension, not first.
        (data, targets) = batch
        data = data.t()
        targets = targets.t()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)

        # Shape of output and targets need to align.
        loss = criterion(output.view(-1, ntokens), targets.flatten())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_loader), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if cumulative_steps is not None:
          cumulative_steps += 1
          if (args.throughput_estimation_interval is not None and
              cumulative_steps % args.throughput_estimation_interval == 0):
              print('[THROUGHPUT_ESTIMATION]\t%s\t%d' % (time.time(),
                                                         cumulative_steps))

          if args.steps is not None and cumulative_steps >= args.steps:
            done = True
            break
        if args.max_duration is not None:
          cumulative_time += time.time() - total_duration_tracker_start
          total_duration_tracker_start = time.time()
          if cumulative_time >= args.max_duration:
            done = True
            break

    return (cumulative_steps, cumulative_time, done)

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    if args.epochs is None:
        args.epochs = args.steps
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        cumulative_steps, cumulative_time, done = train(cumulative_steps,
                                                        cumulative_time)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time)))
        if args.enable_gavel_iterator:
            if train_loader.done:
                break
            elif done:
                train_loader.complete()
                break
        elif done:
          break
        print('-' * 89)
    with open(checkpoint_path, 'wb') as f:
        print('Saving checkpoint at %s...' % (checkpoint_path))
        if args.distributed:
            state = {'model': model.module}
        else:
            state = {'model': model}
        if args.rank == 0 or args.rank is None:
            torch.save(state, f)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
