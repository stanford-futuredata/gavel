import argparse
import multiprocessing
import os
import random
import shutil
import string
import subprocess

NUM_TRAINING_ITERATIONS = 10000

def random_string_generator(n):
  chars = string.ascii_uppercase + string.digits
  return ''.join(random.SystemRandom().choice(chars) for _ in range(n))

def measure_throughput_cnn(args):
  model, data_dir, batch_size, max_duration = args
  num_batches = NUM_TRAINING_ITERATIONS
  cmd_loc = '../workloads/tensorflow/image_classification'
  profile_cmd = ('cd %s; python tf_cnn_benchmarks.py --num_gpus=1 '
                 '--batch_size=%d --num_batches=%d --model=%s --data_dir=%s '
                 '--max_duration=%f --allow_growth') % (cmd_loc, batch_size,
                                                        num_batches, model,
                                                        data_dir, max_duration)
  FNULL = open(os.devnull, 'w')
  try:
    output = subprocess.check_output(profile_cmd,
                                     stderr=FNULL,
                                     shell=True).decode('utf-8')
  except Exception as e:
    return None, e
  for line in output.split('\n'):
    if 'total images/sec:' in line:
      throughput = float(line.split('total images/sec:')[1].strip())
  return throughput, None

def measure_throughput_nmt(args):
  model, data_dir, max_duration = args
  cmd_loc = '../workloads/tensorflow/translation'
  model_dir = '/tmp/nmt_model_%s' % (random_string_generator(6))
  profile_cmd = ('cd %s; python -m nmt.nmt --src=vi --tgt=en '
                 '--vocab_prefix=%s/vocab --train_prefix=%s/train '
                 '--dev_prefix=%s/tst2012 --test_prefix=%s/tst2013 '
                 '--out_dir=%s --num_layers=2 --num_units=128 '
                 '--dropout=0.2 --metrics=bleu '
                 '--max_duration=%f') % (cmd_loc, data_dir, data_dir,
                                         data_dir, data_dir, model_dir,
                                         max_duration)

  FNULL = open(os.devnull, 'w')
  try:
    output = subprocess.check_output(profile_cmd,
                                     stderr=FNULL,
                                     shell=True).decode('utf-8')
  except Exception as e:
    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    return None, e

  output = output.split('\n')[-2]
  wps = float(output[output.index('wps')+4:output.index('ppl')-2])
  if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
  return wps, None

def measure_throughput_lm(args):
  model, data_dir, model_size, max_duration = args
  cmd_loc = '../workloads/tensorflow/language_modeling'
  profile_cmd = ('cd %s; python ptb_word_lm.py --data_path=%s '
                 '--model=%s --max_duration=%f') % (cmd_loc,
                                                    data_dir,
                                                    model_size,
                                                    max_duration)
  FNULL = open(os.devnull, 'w')
  try:
    output = subprocess.check_output(profile_cmd,
                                     stderr=FNULL,
                                     shell=True).decode('utf-8')
  except Exception as e:
    return None, e

  output = output.strip().split('\n')[-1]
  wps = output.split(' ')[-1]
  return float(wps) / 1000.0, None

def measure_throughput_vae(args):
  model, max_duration = args
  cmd_loc = '../workloads/tensorflow/image_generation'
  profile_cmd = ('cd %s; python main.py --working_directory /tmp/gan '
                 ' --model vae --max_duration=%f') % (cmd_loc, max_duration)

  FNULL = open(os.devnull, 'w')
  try:
    output = subprocess.check_output(profile_cmd,
                                     stderr=FNULL,
                                     shell=True).decode('utf-8')
  except Exception as e:
    return None, e

  throughput = float(output.strip().split('\n')[-1].split(':')[1])
  return throughput, None

def measure_throughput(args):
  model = args[0]
  if (model == 'resnet50' or model == 'vgg16' or model == 'inception3' or
      model == 'alexnet'):
    return measure_throughput_cnn(args)
  elif model == 'nmt':
    return measure_throughput_nmt(args)
  elif 'lm' in model:
    return measure_throughput_lm(args)
  elif model == 'vae':
    return measure_throughput_vae(args)
  else:
    return None, 'Invalid model'

def main(output_file, data_dir, minutes, debug):
  config_debug = {
    'resnet50': ('resnet50', data_dir['cnn'], 64, 1),
    'vae': ('vae', 1),
    'nmt': ('nmt', data_dir['nmt'], 1),
    'lm_small': ('lm_small', data_dir['lm'], 'small', 1)
  }

  config = {
      'alexnet': ('alexnet', data_dir['cnn'], 64, minutes),
      'resnet50': ('resnet50', data_dir['cnn'], 64, minutes),
      'vgg16': ('vgg16', data_dir['cnn'], 16, minutes),
      'inception3': ('inception3', data_dir['cnn'], 64, minutes),
      'nmt': ('nmt', data_dir['nmt'], minutes),
      'lm_small': ('lm_small', data_dir['lm'], 'small', minutes),
      'lm_medium': ('lm_medium', data_dir['lm'], 'medium', minutes),
      'lm_large': ('lm_large', data_dir['lm'], 'large', minutes),
      'vae': ('vae', minutes)
  }

  if debug:
    config = config_debug

  models = [model for model in config]
  results = {}

  with open(output_file, 'w') as f:
    for model in models:
      f.write(model + '\n')
      f.flush()
      output, error = measure_throughput(config[model])
      if error is None:
        f.write(str(output) + '\n')
        f.write('\n')
        results[(model, None)] = output
      else:
        raise Exception(error)
    combinations = []
    for i in range(len(models)):
      model1 = models[i]
      for j in range(i, len(models)):
        model2 = models[j]
        combinations.append([config[model1], config[model2]])
    for args in combinations:
      pool = multiprocessing.Pool(2)
      key = (args[0][0], args[1][0])
      f.write('(%s, %s)\n' % (key[0], key[1]))
      f.flush()
      result = []
      try:
        for output, error in pool.imap(measure_throughput, args):
          if error is None:
            result.append(output)
          else:
            result.append(-1)
            print(error)
            continue
      except Exception as e:
        print(e)
        continue
      f.write(str(result) + '\n')
      results[key] = result
      f.write('\n')
      pool.close()
      pool.join()

if __name__=='__main__':
  parser = argparse.ArgumentParser(
      description='Measure throughput of co-located models')
  parser.add_argument('-o', '--output_file', type=str, required=True,
                      help='File to write results to')
  parser.add_argument('-c', '--data_dir_cnn', type=str,
                      default='/home/keshavsanthanam/data/imagenet',
                      help='Data directory for CNNs')
  parser.add_argument('-n', '--data_dir_nmt', type=str,
                      default='/home/keshavsanthanam/data/nmt_data',
                      help='Data directory for NMT')
  parser.add_argument('-r', '--data_dir_lm', type=str,
                      default='/home/keshavsanthanam/data/ptb',
                      help='Data directory for language modeling RNN')
  parser.add_argument('-m', '--minutes', type=int, default=5,
                      help='Number of minutes to run each model for')
  parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                      help='Run in debug mode')

  args = parser.parse_args()
  opt_dict = vars(args)

  output_file = opt_dict['output_file']
  data_dir = {
      'cnn': opt_dict['data_dir_cnn'],
      'nmt': opt_dict['data_dir_nmt'],
      'lm': opt_dict['data_dir_lm']
  }
  debug = opt_dict['debug']
  minutes = opt_dict['minutes']

  main(output_file, data_dir, minutes, debug)
