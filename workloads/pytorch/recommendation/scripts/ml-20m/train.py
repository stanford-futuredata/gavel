import multiprocessing as mp

import argparse
import glog
import math
import pandas as pd
import shutil
import sys
import time
import os
import torch

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.metrics import AveragePrecision, Recall, NDCG
from recoder.nn import DynamicAutoencoder, MatrixFactorization
from recoder.utils import dataframe_to_csr_matrix

parser = argparse.ArgumentParser(description='Recommendation')
parser.add_argument('-n', '--num_epochs', required=True, type=int,
                    help='Number of epochs to run for')
parser.add_argument('--throughput_estimation_interval', type=int, default=None,
                    help='Steps between logging steps completed')
parser.add_argument('-d', '--data_dir', required=True, type=str,
                    help='Data directory')
parser.add_argument('-b', '--batch_size', default=2048, type=int,
                    help='Batch size')
parser.add_argument('--checkpoint_dir', type=str,
                    default='/lfs/1/keshav2/checkpoints/recommendation',
                    help='Checkpoint dir')
args = parser.parse_args()

data_dir = args.data_dir #'/home/keshavsanthanam/data/ml-20m/pro_sg/'
#model_dir = '/tmp/models/ml-20m/'

if not os.path.isdir(args.checkpoint_dir):
  os.makedirs(args.checkpoint_dir)
checkpoint_path = os.path.join(args.checkpoint_dir, 'model.chkpt')

common_params = {
  'user_col': 'uid',
  'item_col': 'sid',
  'inter_col': 'watched',
}

glog.info('Loading Data...')

train_df = pd.read_csv(data_dir + 'train.csv')
val_tr_df = pd.read_csv(data_dir + 'validation_tr.csv')
val_te_df = pd.read_csv(data_dir + 'validation_te.csv')

# uncomment it to train with MatrixFactorization
# train_df = train_df.append(val_tr_df)

train_matrix, item_id_map, _ = dataframe_to_csr_matrix(train_df, **common_params)
val_tr_matrix, _, user_id_map = dataframe_to_csr_matrix(val_tr_df, item_id_map=item_id_map,
                                                        **common_params)
val_te_matrix, _, _ = dataframe_to_csr_matrix(val_te_df, item_id_map=item_id_map,
                                              user_id_map=user_id_map, **common_params)

train_dataset = RecommendationDataset(train_matrix)
val_tr_dataset = RecommendationDataset(val_tr_matrix, val_te_matrix)


use_cuda = True

model = DynamicAutoencoder(hidden_layers=[200], activation_type='tanh',
                           noise_prob=0.5, sparse=False)

# model = MatrixFactorization(embedding_size=200, activation_type='tanh',
#                             dropout_prob=0.5, sparse=False)

trainer = Recoder(model=model, use_cuda=use_cuda, optimizer_type='adam',
                  loss='logistic', user_based=False)
if os.path.exists(checkpoint_path):
    print('Loading checkpoint from %s...' % (checkpoint_path))
    trainer.init_from_model_file(checkpoint_path)

metrics = [Recall(k=20, normalize=True), Recall(k=50, normalize=True),
           NDCG(k=100)]

try:
  if args.throughput_estimation_interval is not None:
      num_iterations = int(math.ceil(args.num_epochs / args.throughput_estimation_interval))
      epochs_per_iteration = args.throughput_estimation_interval
  else:
      num_iterations = 1
      epochs_per_iteration = args.num_epochs
  epochs = 0
  for i in range(num_iterations):
      epochs_per_iteration = min(epochs_per_iteration, args.num_epochs - epochs)
      print('Running for %d epochs' % (epochs_per_iteration))
      #trainer = Recoder(model=model, use_cuda=use_cuda, optimizer_type='adam',
      #                loss='logistic', user_based=False)
      trainer.train(train_dataset=train_dataset, val_dataset=val_tr_dataset,
                    batch_size=args.batch_size, lr=1e-3, weight_decay=2e-5,
                    num_epochs=epochs_per_iteration, negative_sampling=True,
                    lr_milestones=[60, 80], num_data_workers=mp.cpu_count() if use_cuda else 0,
                    model_checkpoint_prefix=None,
                    checkpoint_freq=0, eval_num_recommendations=0,
                    metrics=metrics, eval_freq=0)
      epochs += epochs_per_iteration
      if args.throughput_estimation_interval is not None:
            print('[THROUGHPUT_ESTIMATION]\t%s\t%d' % (time.time(), epochs))
            #shutil.rmtree(model_dir)
            #for f in os.listdir(model_dir):
            #    os.remove(os.path.join(model_dir, f))
            #os.makedirs(model_d)
  current_state = {
      'model_params': trainer.model.model_params(),
      'last_epoch': trainer.current_epoch,
      'model': trainer.model.state_dict(),
      'optimizer_type': trainer.optimizer_type,
      'optimizer': trainer.optimizer.state_dict(),
      'items': trainer.items,
      'users': trainer.users,
      'num_items': trainer.num_items,
      'num_users': trainer.num_users
  }

  if type(trainer.loss) is str:
    current_state['loss'] = trainer.loss
    current_state['loss_params'] = trainer.loss_params

  print('Saving checkpoint at %s...' % (checkpoint_path))
  torch.save(current_state, checkpoint_path)

except (KeyboardInterrupt, SystemExit) as e:
  print(e)
  sys.stdout.flush()
  raise
