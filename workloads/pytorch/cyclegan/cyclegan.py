import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# TODO: Figure out a cleaner way of including gavel_iterator.
cyclegan_dir = os.path.dirname(os.path.realpath(__file__))
pytorch_dir = os.path.dirname(cyclegan_dir)
workloads_dir = os.path.dirname(pytorch_dir)
gpusched_dir = os.path.dirname(workloads_dir)
scheduler_dir = os.path.join(gpusched_dir, 'scheduler')
sys.path.append(scheduler_dir)
from gavel_iterator import GavelIterator

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default="/lfs/1/keshav2/checkpoints/cyclegan", help="Checkpoint dir")
parser.add_argument("--n_steps", type=int, default=None, help="number of steps of training")
parser.add_argument("--n_epochs", type=int, default=None, help="number of epochs of training")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="Name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument('--throughput_estimation_interval', type=int, default=None,
                    help='Steps between logging steps completed')
parser.add_argument('--max_duration', type=int, default=None,
                    help='Maximum duration in seconds')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Local rank')
parser.add_argument('--enable_gavel_iterator', action='store_true',
                    default=False, help='If set, use Gavel iterator')
opt = parser.parse_args()
print(opt)

if opt.n_steps is not None and opt.n_epochs is not None:
  raise ValueError('Only one of n_steps and n_epochs can be set')
elif opt.n_steps is None and opt.n_epochs is None:
  raise ValueError('One of n_steps and n_epochs must be set')

torch.cuda.set_device(opt.local_rank)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
if not os.path.isdir(opt.checkpoint_dir):
    os.mkdir(opt.checkpoint_dir)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset(opt.dataset_path, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset(opt.dataset_path, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

def load_checkpoint(opt, checkpoint_path):
    try:
        print('Loading checkpoint from %s...' % (checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(opt.local_rank))
        return checkpoint
    except Exception as e:
        print('Could not load from checkpoint: %s' % (e))
        return None

def save_checkpoint(state, checkpoint_path):
    print('Saving checkpoint at %s...' % (checkpoint_path))
    torch.save(state, checkpoint_path)

if opt.enable_gavel_iterator:
    dataloader = GavelIterator(dataloader, opt.checkpoint_dir, load_checkpoint, save_checkpoint)

checkpoint_path = os.path.join(opt.checkpoint_dir, "model.chkpt")
checkpoint = None
if os.path.exists(checkpoint_path):
    if opt.enable_gavel_iterator:
        checkpoint = dataloader.load_checkpoint(opt, checkpoint_path)
    else:
        checkpoint = load_checkpoint(opt, checkpoint_path)
else:
    print('Could not load from checkpoint!')

if checkpoint is not None:
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_BA.load_state_dict(checkpoint['G_BA'])
    D_A.load_state_dict(checkpoint['D_A'])
    D_B.load_state_dict(checkpoint['D_B'])
    start_epoch = checkpoint['epoch']
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    start_epoch = 0

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


if opt.n_steps is not None:
    opt.n_epochs = math.ceil(opt.n_steps * opt.batch_size / len(dataloader))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, start_epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, start_epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, start_epoch, opt.decay_epoch).step
)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


# ----------
#  Training
# ----------

done = False
steps = 0
elapsed_time = 0
prev_time = time.time()
for epoch in range(start_epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        if opt.n_steps is not None and steps >= opt.n_steps:
            done = True
            break
        elif opt.max_duration is not None and elapsed_time >= opt.max_duration:
            done = True
            break

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        elapsed_time += time.time() - prev_time
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch+1,
                opt.n_epochs,
                i,
                len(dataloader) if epoch < opt.n_epochs-1 else opt.n_steps - epoch * len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

        steps += 1

        if (opt.throughput_estimation_interval is not None and
            steps % opt.throughput_estimation_interval == 0):
            print('')
            sys.stdout.flush()
            print('[THROUGHPUT_ESTIMATION]\t%s\t%d' % (time.time(), steps))
    if done or (opt.enable_gavel_iterator and dataloader.done):
        break

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

if opt.enable_gavel_iterator:
    dataloader.complete()

state = {
    'G_AB': G_AB.state_dict(),
    'G_BA': G_BA.state_dict(),
    'D_A': D_A.state_dict(),
    'D_B': D_B.state_dict(),
    'epoch': epoch
}
print('')
if opt.enable_gavel_iterator:
    dataloader.save_checkpoint(state, checkpoint_path)
else:
    save_checkpoint(state, checkpoint_path)
