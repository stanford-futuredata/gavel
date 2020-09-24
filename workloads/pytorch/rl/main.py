from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import dill
import torch
import torch.multiprocessing as mp
from environment import atari_env
from utils import read_config
from model import A3Clstm
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
#from gym.configuration import undo_logger_setup
import sys
import time

# TODO: Figure out a cleaner way of including gavel_iterator.
rl_dir = os.path.dirname(os.path.realpath(__file__))
pytorch_dir = os.path.dirname(rl_dir)
workloads_dir = os.path.dirname(pytorch_dir)
gpusched_dir = os.path.dirname(workloads_dir)
scheduler_dir = os.path.join(gpusched_dir, 'scheduler')
sys.path.append(scheduler_dir)
from gavel_iterator import GavelIterator

INFINITY = 1000000

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=32,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default='/lfs/1/keshav2/checkpoints/a3c',
    help='Checkpoint dir')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--max-steps',
    type=int,
    default=None,
    metavar='MS',
    help='Maximum number of steps')
parser.add_argument(
    '--throughput_estimation_interval',
    type=int,
    default=None,
    help='Steps between logging steps completed')
parser.add_argument(
    '--max_duration',
    type=int,
    default=None,
    help='Maximum duration in seconds')
parser.add_argument('--local_rank',
        default=0,
        type=int,
        help='Local rank')
parser.add_argument('--enable_gavel_iterator',
        action='store_true',
        default=False,
        help='If set, use Gavel iterator')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

def load_checkpoint(args, checkpoint_path):
    try:
        print('Loading checkpoint from %s...' % (checkpoint_path))
        return torch.load(checkpoint_path, map_location='cuda:{}'.format(args.local_rank))
    except Exception as e:
        print('Could not load from checkpoint: %s' % (e))
        return None

def save_checkpoint(state, checkpoint_path):
    import torch
    print('Saving checkpoint at %s...' % (checkpoint_path))
    torch.save(state, checkpoint_path)

if __name__ == '__main__':
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    torch.manual_seed(args.seed)
    args.gpu_ids = [args.local_rank]
    torch.cuda.manual_seed(args.seed)
    mp.set_start_method('spawn')
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)

    iters = {}
    for rank in range(0, args.workers):
        iters[rank] = range(args.max_steps)
        if args.enable_gavel_iterator and rank == 0:
            iters[rank] = GavelIterator(iters[rank], args.checkpoint_dir,
                                        load_checkpoint, save_checkpoint,
                                        write_on_close=False)

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.chkpt')
    if os.path.exists(checkpoint_path):
        if args.enable_gavel_iterator:
            saved_state = iters[0].load_checkpoint(args, checkpoint_path)
        else:
            saved_state = load_checkpoint(args, checkpoint_path)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    args.distributed = False

    if args.max_steps is None:
        args.max_steps = INFINITY

    processes = []

    time.sleep(0.1)
    for rank in range(0, args.workers):
        if args.enable_gavel_iterator and rank == 0:
            iters[rank]._close_file_handler()
        iters_ = dill.dumps(iters[rank])
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf,
                                iters_, checkpoint_path))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    processes[0].join()
    if len(processes) > 1:
        for p in processes[1:]:
            p.terminate()
            p.join()
    if not args.enable_gavel_iterator:
        state = shared_model.state_dict()
        save_checkpoint(state, checkpoint_path)
