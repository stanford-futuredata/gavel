from __future__ import print_function

import argparse
import multiprocessing
import os
import random
import shutil
import string
import subprocess
import sys

metrics = [
  'per_kernel_runtimes',
  'flop_count_sp',
  'sm_efficiency',
  'dram_read_throughput',
  'dram_write_throughput',
  'eligible_warps_per_cycle',
  'flop_sp_efficiency',
  'ipc',
  'l2_read_throughput',
  'l2_write_throughput',
  'warp_execution_efficiency',
]

def random_string_generator(n):
  chars = string.ascii_uppercase + string.digits
  return ''.join(random.SystemRandom().choice(chars) for _ in range(n))

def measure_metric_cnn(args):
  model, data_dir, batch_size, num_batches, metric = args
  cmd_loc = '../workloads/tensorflow/image_classification'
  model_cmd = ('python tf_cnn_benchmarks.py --num_gpus=1 '
               '--batch_size=%d --num_batches=%d --num_warmup_batches=0 '
               '--model=%s --data_dir=%s --allow_growth') % (batch_size,
                                                             num_batches,
                                                             model, data_dir)
  profile_cmd = ('cd %s; /usr/local/cuda/bin/nvprof %s --csv --devices 0 '
                 '--normalized-time-unit ms %s') % (cmd_loc, metric, model_cmd)
  try:
    output = subprocess.check_output(profile_cmd,
                                     stderr=subprocess.STDOUT,
                                     shell=True).decode('utf-8')
  except Exception as e:
    return None, e

  return output, None

def measure_metric_nmt(args):
  model, data_dir, num_train_steps, metric = args
  cmd_loc = '../workloads/tensorflow/translation'
  model_dir = '/tmp/nmt_model_%s' % (random_string_generator(6))
  model_cmd = ('python -m nmt.nmt --src=vi --tgt=en '
               '--vocab_prefix=%s/vocab --train_prefix=%s/train '
               '--dev_prefix=%s/tst2012 --test_prefix=%s/tst2013 '
               '--out_dir=%s --num_layers=2 --num_units=128 '
               '--dropout=0.2 --metrics=bleu '
               '--num_train_steps=%d') % (data_dir, data_dir, data_dir,
                                          data_dir, model_dir, num_train_steps)
  profile_cmd = ('cd %s; /usr/local/cuda/bin/nvprof %s --csv --devices 0 '
                 '--normalized-time-unit ms %s') % (cmd_loc, metric, model_cmd)

  try:
    output = subprocess.check_output(profile_cmd,
                                     stderr=subprocess.STDOUT,
                                     shell=True).decode('utf-8')
  except Exception as e:
    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    return None, e

  return output, None

def measure_metric_lm(args):
  model, data_dir, model_size, max_steps, metric = args
  cmd_loc = '../workloads/tensorflow/language_modeling'
  model_cmd = ('python ptb_word_lm.py --data_path=%s --model=%s '
               '--max_max_epoch=1 --max_steps=%d') % (data_dir, model_size,
                                                      max_steps)
  profile_cmd = ('cd %s; /usr/local/cuda/bin/nvprof %s --csv --devices 0 '
                 '--normalized-time-unit ms %s') % (cmd_loc, metric, model_cmd)
  try:
    output = subprocess.check_output(profile_cmd,
                                     stderr=subprocess.STDOUT,
                                     shell=True).decode('utf-8')
  except Exception as e:
    return None, e

  return output, None

def measure_metric_vae(args):
  model, updates_per_epoch, metric = args
  cmd_loc = '../workloads/tensorflow/image_generation'
  model_cmd = ('python main.py --working_directory /tmp/gan '
               ' --model vae --max_epoch 1 '
               '--updates_per_epoch %d') % (updates_per_epoch)
  profile_cmd = ('cd %s; /usr/local/cuda/bin/nvprof %s --csv --devices 0 '
                 '--normalized-time-unit ms %s') % (cmd_loc, metric, model_cmd)

  try:
    output = subprocess.check_output(profile_cmd,
                                     stderr=subprocess.STDOUT,
                                     shell=True).decode('utf-8')
  except Exception as e:
    return None, e

  return output, None

def measure_metric(args):
  model = args[0]

  if (model == 'resnet50' or model == 'vgg16' or model == 'inception3' or
      model == 'alexnet'):
    return measure_metric_cnn(args)
  elif model == 'nmt':
    return measure_metric_nmt(args)
  elif 'lm' in model:
    return measure_metric_lm(args)
  elif model == 'vae':
    return measure_metric_vae(args)
  else:
    return None, 'Invalid model'

def main(log_dir, data_dir, num_steps, debug):
  config_debug = {
    'nmt': ('nmt', data_dir['nmt'], 1),
    'lm_small': ('lm_small', data_dir['lm'], 'small', 2),
    'resnet50': ('resnet50', data_dir['cnn'], 64, 2),
    'vae': ('vae', 1),
  }

  config = {
    'alexnet': ('alexnet', data_dir['cnn'], 64, num_steps),
    'resnet50': ('resnet50', data_dir['cnn'], 64, num_steps),
    'vgg16': ('vgg16', data_dir['cnn'], 16, num_steps),
    'inception3': ('inception3', data_dir['cnn'], 64, num_steps),
    'nmt': ('nmt', data_dir['nmt'], num_steps),
    'lm_small': ('lm_small', data_dir['lm'], 'small', num_steps),
    'lm_medium': ('lm_medium', data_dir['lm'], 'medium', num_steps),
    'lm_large': ('lm_large', data_dir['lm'], 'large', num_steps),
    'vae': ('vae', num_steps)
  }

  if debug:
    config = config_debug

  models = [model for model in config]
  results = {}

  if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

  for model in models:
    model_path = os.path.join(log_dir, model)
    if not os.path.isdir(model_path):
      os.makedirs(model_path)
    for metric in metrics:
      if metric == 'per_kernel_runtimes':
        output, error = measure_metric(config[model] + ('',))
      else:
        output, error = measure_metric(config[model]
            + ('--metrics %s' % (metric),))
      if output is not None:
        log_path = os.path.join(model_path, metric + '.csv')
        with open(log_path, 'w') as f:
          f.write(output)
      elif error is not None:
        print(error, file=sys.stderr)
        continue
      else:
        sys.exit('No information returned for metric %s,'
                 'model %s' % (metric, model))

if __name__=='__main__':
  parser = argparse.ArgumentParser(
      description='Measure nvprof metrics for different models')
  parser.add_argument('-l', '--log_dir', type=str, required=True,
                      help='Log directory')
  parser.add_argument('-c', '--data_dir_cnn', type=str,
                      default='/home/keshavsanthanam/data/imagenet',
                      help='Data directory for CNNs')
  parser.add_argument('-n', '--data_dir_nmt', type=str,
                      default='/home/keshavsanthanam/data/nmt_data',
                      help='Data directory for NMT')
  parser.add_argument('-r', '--data_dir_lm', type=str,
                      default='/home/keshavsanthanam/data/ptb',
                      help='Data directory for language modeling RNN')
  parser.add_argument('-m', '--num_steps', type=int, default=10,
                      help='Number of steps to run each model for')
  parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                      help='Run in debug mode')

  args = parser.parse_args()
  opt_dict = vars(args)

  log_dir = opt_dict['log_dir']
  data_dir = {
      'cnn': opt_dict['data_dir_cnn'],
      'nmt': opt_dict['data_dir_nmt'],
      'lm': opt_dict['data_dir_lm']
  }
  debug = opt_dict['debug']
  num_steps = opt_dict['num_steps']

  main(log_dir, data_dir, num_steps, debug)
