from __future__ import print_function

import argparse
import csv
import sys

model_names = {
  'alexnet': 'AlexNet',
  'inception3': 'Inception v3',
  'lm_large': 'Language Model (large)',
  'lm_small': 'Language Model (small)',
  'lm_medium': 'Language Model (medium)',
  'vae': 'VAE',
  'nmt': 'NMT',
  'resnet50': 'ResNet-50',
  'vgg16': 'VGG16',
}

def parse(runtimes):
  runtimes = runtimes.replace('[', '').replace(']', '')
  times = runtimes.split(',')
  if float(times[0]) == -1 or float(times[1]) == -1:
    return '0, 0'
  else:
    return '%.2f, %.2f' % (float(times[0]), float(times[1]))

def flip(runtimes):
  times = runtimes.split(',')
  return '%.2f, %.2f' % (float(times[1]), float(times[0]))

def to_ratio(combination_runtimes, model1_runtime, model2_runtime):
  times = combination_runtimes.split(',')
  return '%.2f, %.2f' % (float(times[0]) / float(model1_runtime),
                         float(times[1]) / float(model2_runtime))

def get_runtimes(path):
  runtimes = {}
  with open(path, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    for i in range(0, len(lines), 3):
      runtimes[lines[i]] = lines[i+1]
  models = []
  for combination in runtimes:
    if ',' not in combination:
      models.append(combination)
  models = sorted(models)
  return runtimes, models

def process_log_file(runtimes, models, ratios=False):
  sys.stdout.write('||')
  for model in models:
    sys.stdout.write(model_names[model] + '|')
  print('')
  for i in range(len(models)):
    model1 = models[i]
    sys.stdout.write(model_names[model1] + '|' + runtimes[model1] + '|')
    for j in range(len(models)):
      if j < i:
        sys.stdout.write('|')
      else:
        model2 = models[j]
        combination = '(%s, %s)' % (model1, model2)
        if combination in runtimes:
          if ratios:
            sys.stdout.write(to_ratio(parse(runtimes[combination]),
                                      runtimes[model1], runtimes[model2])+ '|')
          else:
            sys.stdout.write(parse(runtimes[combination]) + '|')
          continue
        combination = '(%s, %s)' % (model2, model1)
        if combination in runtimes:
          if ratios:
            sys.stdout.write(to_ratio(flip(parse(runtimes[combination])),
                                      runtimes[model1], runtimes[model2])+ '|')
          else:
            sys.stdout.write(parse(runtimes[combination]) + '|')
          continue
        sys.exit('Could not find runtime of %s' % combination)
    print('')

def main(log_file):
  runtimes, models = get_runtimes(log_file)
  process_log_file(runtimes, models)
  process_log_file(runtimes, models, ratios=True)

if __name__=='__main__':
  parser = argparse.ArgumentParser(
      description='Parse scheduling co-location logs')
  parser.add_argument('-f', '--file', type=str, required=True, help='Log file')

  args = parser.parse_args()
  opt_dict = vars(args)

  log_file = opt_dict['file']

  main(log_file)
