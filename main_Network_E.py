# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:30:03 2018

@author: hsseo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from batch import *

from Network_E import *



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

      for lindex in range(1,1):
          alpha_p_valu_keep = alpha_p_value
          alpha_n_valu_keep = alpha_n_value
          beta_sq_valu_keep = beta_sq_value
          
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result1 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error1 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result1))
          me1 = tf.reduce_mean(ms_error1)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result2 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error2 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result2))
          me2 = tf.reduce_mean(ms_error2)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result3 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error3 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result3))
          me3 = tf.reduce_mean(ms_error3)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result4 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error4 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result4))
          me4 = tf.reduce_mean(ms_error4)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result5 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error5 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result5))
          me5 = tf.reduce_mean(ms_error5)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result6 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error6 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result6))
          me6 = tf.reduce_mean(ms_error6)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result7 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error7 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result7))
          me7 = tf.reduce_mean(ms_error7)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result8 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error8 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result8))
          me8 = tf.reduce_mean(ms_error8)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result9 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error9 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result9))
          me9 = tf.reduce_mean(ms_error9)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result10 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error10 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result10))
          me10 = tf.reduce_mean(ms_error10)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result11 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error11 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result11))
          me11 = tf.reduce_mean(ms_error11)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result12 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error12 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result12))
          me12 = tf.reduce_mean(ms_error12)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result13 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error13 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result13))
          me13 = tf.reduce_mean(ms_error13)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result14 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error14 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result14))
          me14 = tf.reduce_mean(ms_error14)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result15 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error15 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result15))
          me15 = tf.reduce_mean(ms_error15)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result16 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error16 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result16))
          me16 = tf.reduce_mean(ms_error16)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result17 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error17 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result17))
          me17 = tf.reduce_mean(ms_error17)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result18 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error18 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result18))
          me18 = tf.reduce_mean(ms_error18)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result19 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error19 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result19))
          me19 = tf.reduce_mean(ms_error19)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.0
          
          _, temp_result20 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error20 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result20))
          me20 = tf.reduce_mean(ms_error20)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result21 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error21 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result21))
          me21 = tf.reduce_mean(ms_error21)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result22 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error22 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result22))
          me22 = tf.reduce_mean(ms_error22)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result23 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error23 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result23))
          me23 = tf.reduce_mean(ms_error23)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result24 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error24 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result24))
          me24 = tf.reduce_mean(ms_error24)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result25 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error25 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result25))
          me25 = tf.reduce_mean(ms_error25)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.1
          
          _, temp_result26 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error26 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result26))
          me26 = tf.reduce_mean(ms_error26)
          ##
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 0.9
          
          _, temp_result27 = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          ms_error27 = tf.losses.mean_squared_error(tf.squeeze(batch_Y), tf.squeeze(temp_result27))
          me27 = tf.reduce_mean(ms_error27)
          ##
          
          me_mat = [me1, me2, me3, me4, me5, me6, me7, me8, mr9, me10, me11, me12, me13, me14, me15, me16, me17, me18, me19, me20, me21, me22, me23, me24, me25, me26, me27]
          if tf.argmin(me_mat) == 0:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 1:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 2:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 3:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          elif tf.armin(me_mat) == 4:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 5:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          elif tf.armin(me_mat) == 6:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          elif tf.armin(me_mat) == 7:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 8:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 9:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 0.9
          
          elif tf.armin(me_mat) == 10:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 0.9
          
          elif tf.armin(me_mat) == 11:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 12:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 0.9
          
          elif tf.armin(me_mat) == 13:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 0.9
          
          elif tf.armin(me_mat) == 14:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 0.9
          
          elif tf.armin(me_mat) == 15:
             alpha_p_value = alpha_p_value_keep* 1.0
             alpha_n_value = alpha_n_value_keep* 1.1
             beta_sq_value = beta_sq_value_keep* 0.9
          
          elif tf.armin(me_mat) == 16:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          elif tf.armin(me_mat) == 17:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          elif tf.armin(me_mat) == 18:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 19:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.0
          
          elif tf.armin(me_mat) == 20:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          elif tf.armin(me_mat) == 21:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          elif tf.armin(me_mat) == 22:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          elif tf.armin(me_mat) == 23:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 0.9
          
          elif tf.armin(me_mat) == 24:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 0.9
          
          elif tf.armin(me_mat) == 25:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.1
          
          else:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 0.9
          

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
         print('alpha_p_value %f' % (alpha_p_value))
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
