# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:30:03 2018

@author: hsseo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from batch import *

from Network_D import *



# pylint: disable=missing-docstring
import argparse
import os.path
import os
import sys
import time
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import scipy.io as sio
import numpy as np

#tf.device('/cpu:0'):
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
#os.environ["CUDA_lISIBLE_DEVICES"]="2,3"

image_size = 512
# Basic model parameters as external flags.
FLAGS = None



X = sio.loadmat('/raid/seo/CT/lung/TrainInput_l1.mat')
X1 = X['TrainInput_l1']

X_train_s = np.concatenate([X1],axis=0)

Y = sio.loadmat('/raid/seo/CT/lung/TrainOutput_l1.mat')
Y1 = Y['TrainOutput_l1']

Y_train_s = np.concatenate([Y1],axis=0)


X_t = sio.loadmat('/raid/seo/CT/lung/TestInput_l1.mat')
X_test = X_t['TestInput_l1']
Y_t = sio.loadmat('/raid/seo/CT/lung/TestOutput_l1.mat')
Y_test = Y_t['TestOutput_l1']




def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,512,512))
  labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size,512,512))
  return images_placeholder,labels_placeholder


def run_training(): 
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
  
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32)
    
    # Build a Graph that computes predictions from the inference model.
    logits = deepnn(images_placeholder, image_size, FLAGS.batch_size, keep_prob, phase_train)
    
   
    alpha_p = tf.placeholder(tf.float32, shape=())
    alpha_n = tf.placeholder(tf.float32, shape=())
    beta_sq = tf.placeholder(tf.float32, shape=())

    
    # Add to the Graph the Ops for loss calculation.
    loss, updated_alpha_p, updated_alpha_n, updated_beta_sq = lossfn(logits, labels_placeholder,alpha_p, alpha_n, beta_sq)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, FLAGS.learning_rate)

    
    # calculate prediction error
    #pred_err = prediction(logits, labels_placeholder, labels_mean, labels_std)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()
    #init = tf.initialize_all_variables

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)
    

    TRAIN_DATASIZE = X_train_s.shape[0]
    batchtrain = Batchdata(np.arange(0,TRAIN_DATASIZE))
    
    alpha_p_value = 1.0    
    alpha_n_value = 1.0 
    beta_sq_value = 1.0
    for step in xrange(FLAGS.max_step):
        
      start_time = time.time()

      idxs = batchtrain.next_batch(FLAGS.batch_size) #shuffled ordering
      batch_X = X_train_s[idxs,:,:]
      batch_Y = Y_train_s[idxs,:,:]
              
      feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 0.6, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value}
    
    
      _, loss_value, alpha_p_value, alpha_n_value, beta_sq_value = sess.run([train_op, loss, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)    
    
    
      duration = time.time() - start_time    
    

      if (step + 1) % 5 == 0:
         checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
         saver.save(sess, checkpoint_file, global_step=step)
         print('%%%% save the model paramters ... ')
         print('Step %d: loss = %.7f (%.3f sec)' % (step, loss_value, duration))
        
      if (step + 1) % 5 == 0:
         feed_dict = {images_placeholder: X_test, labels_placeholder: Y_test, keep_prob: 1.0, phase_train: False, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
         Test_results = sess.run([logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
                  
         loss_value = sess.run(loss, feed_dict=feed_dict)    
         print('alpha_p2_value %f' % (alpha_p_value))
         print('alpha_n_value %f' % (alpha_n_value))
         print('beta_sq_value %f' % (beta_sq_value))
         
         sio.savemat('Test_results.mat', {'pred': Test_results})
  
    
    
    
################################################################
#####################run_kernel#################################
################################################################

def main(_):
  
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.00001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_step',
      type=int,
      default=2000000000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=10,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='logs',
      help='Directory to put the log data.'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
