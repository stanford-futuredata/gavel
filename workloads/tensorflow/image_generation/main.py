'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import datetime
import time

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

from progressbar import ETA, Bar, Percentage, ProgressBar

from vae import VAE
from gan import GAN

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "gan or vae")
flags.DEFINE_float("max_duration", None,
                   "Maximum training duration in minutes") 

FLAGS = flags.FLAGS

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=True)

    assert FLAGS.model in ['vae', 'gan']
    if FLAGS.model == 'vae':
        model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)
    elif FLAGS.model == 'gan':
        model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)
      
    start_time_mb = datetime.datetime.now()
    total_images = 0
    done = False
    print("start:", start_time_mb)
    epoch = 0
    while epoch < FLAGS.max_epoch or not done:
        training_loss = 0.0

        pbar = ProgressBar()
        for i in pbar(range(FLAGS.updates_per_epoch)):
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
            loss_value = model.update_params(images)
            training_loss += loss_value
            total_images += FLAGS.batch_size
            seconds = (datetime.datetime.now() - start_time_mb).total_seconds()
            minutes = seconds / 60.0
            if FLAGS.max_duration is not None and minutes >= FLAGS.max_duration:
              end_time_mb = datetime.datetime.now()
              print("end:", str(end_time_mb))
              total_time = (end_time_mb - start_time_mb).total_seconds()
              total_throughput = total_images / float(total_time)
              print('Total images/second: %.f' % (total_throughput)) 
              done = True
              break
        if done == True:
          break

        training_loss = training_loss / \
            (FLAGS.updates_per_epoch * FLAGS.batch_size)

        print("Loss %f" % training_loss)

        model.generate_and_save_images(
            FLAGS.batch_size, FLAGS.working_directory)
        
        epoch += 1
        if FLAGS.max_duration is None and epoch == FLAGS.max_epoch:
          break
