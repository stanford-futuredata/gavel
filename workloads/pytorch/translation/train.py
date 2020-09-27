'''
This script handling the training process.
'''

import argparse
import glob
import math
import os
import sys
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

# TODO: Figure out a cleaner way of including gavel_iterator.
translation_dir = os.path.dirname(os.path.realpath(__file__))
pytorch_dir = os.path.dirname(translation_dir)
workloads_dir = os.path.dirname(pytorch_dir)
gpusched_dir = os.path.dirname(workloads_dir)
scheduler_dir = os.path.join(gpusched_dir, 'scheduler')
sys.path.append(scheduler_dir)
from gavel_iterator import GavelIterator


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing,
                step=None, max_duration=None, cumulative_step=None,
                cumulative_time=None, throughput_estimation_interval=None):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    done = False

    start_time = time.time()
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

        if cumulative_step is not None:
          cumulative_step += 1

          if (throughput_estimation_interval is not None and
              cumulative_step % throughput_estimation_interval == 0):
              print('[THROUGHPUT_ESTIMATION]\t%s\t%d' % (time.time(),
                                                         cumulative_step))
          if step is not None and cumulative_step >= step:
              done = True
              break
        if cumulative_time is not None:
          cumulative_time += time.time() - start_time
          start_time = time.time()

          if max_duration is not None and cumulative_time >= max_duration:
            done = True
            break

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy, cumulative_step, cumulative_time, done

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def load_checkpoint(opt, checkpoint_path):
    print('Loading checkpoint from %s...' % (checkpoint_path))
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(opt.local_rank))
        return checkpoint
    except Exception as e:
        print('Could not load from checkpoint: %s' % (e))
        return None

def save_checkpoint(state, checkpoint_path):
    print('Saving checkpoint at %s...' % (checkpoint_path))
    torch.save(state, checkpoint_path)

def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    start_epoch = 0
    if not os.path.isdir(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)
    checkpoint_path = os.path.join(opt.checkpoint_dir, 'model.chkpt')
    if opt.enable_gavel_iterator:
        checkpoint = training_data.load_checkpoint(opt, checkpoint_path)
    else:
        checkpoint = load_checkpoint(opt, checkpoint_path)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        print('No checkpoint file found!')

    valid_accus = []
    if opt.epoch is None:
        opt.epoch = opt.step
    cumulative_step = 0
    cumulative_time = 0

    if opt.step is not None:
        opt.epoch = math.ceil(float(opt.step) *
                              opt.batch_size / len(training_data))

    for epoch_i in range(start_epoch, opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, cumulative_step, cumulative_time, done =\
                train_epoch(model, training_data, optimizer, device,
                            smoothing=opt.label_smoothing,
                            step=opt.step,
                            max_duration=opt.max_duration,
                            cumulative_step=cumulative_step,
                            cumulative_time=cumulative_time,
                            throughput_estimation_interval=opt.throughput_estimation_interval)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))
        cumulative_time += time.time() - start
        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i,
        }

        if log_train_file:#and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                """
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
                """
        if opt.enable_gavel_iterator:
            if training_data.done:
                break
            elif done:
                # Early stop.
                training_data.complete()
                break
        elif done:
            break

    if not opt.distributed or opt.rank == 0:
        if opt.save_mode == 'all':
            if opt.enable_gavel_iterator:
                training_data.save_checkpoint(checkpoint, checkpoint_path)
            else:
                save_checkpoint(checkpoint, checkpoint_path)
        elif opt.save_mode == 'best':
            if valid_accu >= max(valid_accus):
                print('Saving checkpoint at %s...' % (checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                print('    - [Info] The checkpoint file has been updated.')


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=None)
    parser.add_argument('-step', type=int, default=None)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    # NOTE(keshav2): This just refers to the learning rate schedule,
    #                nothing performance related.
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/lfs/1/keshav2/checkpoints/transformer')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

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

    parser.add_argument('--throughput_estimation_interval', type=int,
                        default=None,
                        help='Steps between logging steps completed')
    parser.add_argument('--max_duration', type=int, default=None,
                        help='Maximum duration in seconds')
    parser.add_argument('--enable_gavel_iterator', action='store_true',
                        default=False, help='If set, use Gavel iterator')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    torch.cuda.set_device(opt.local_rank)

    if opt.epoch is not None and opt.step is not None:
        raise ValueError('Only one of epoch and step may be set')
    elif opt.epoch is None and opt.step is None:
        raise ValueError('One of epoch and step must be set')

    opt.distributed = False
    if opt.master_addr is not None:
        opt.distributed = True
        os.environ['MASTER_ADDR'] = opt.master_addr
        os.environ['MASTER_PORT'] = str(opt.master_port)
        dist.init_process_group(backend=opt.dist_backend,
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.rank)

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt, opt.master_addr is not None)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    if opt.distributed:
        transformer = DDP(transformer, device_ids=[opt.local_rank],
                          output_device=opt.local_rank)

    if opt.enable_gavel_iterator:
        training_data = GavelIterator(training_data, opt.checkpoint_dir,
                                      load_checkpoint, save_checkpoint)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)


def prepare_dataloaders(data, opt, distributed):
    # ========= Preparing DataLoader =========#
    train_dataset = TranslationDataset(
        src_word2idx=data['dict']['src'],
        tgt_word2idx=data['dict']['tgt'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'])
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=train_sampler is None,
        sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
