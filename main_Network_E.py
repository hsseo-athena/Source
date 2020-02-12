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

      for lindex in range(1,2):
          alpha_p_valu_keep = alpha_p_value
          alpha_n_valu_keep = alpha_n_value
          beta_sq_valu_keep = beta_sq_value
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.0
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result1,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me1 = ((batch_Y-np.squeeze(temp_result1))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.0
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result2,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me2 = ((batch_Y-np.squeeze(temp_result2))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.0
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result3,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me3 = ((batch_Y-np.squeeze(temp_result3))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.1
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result4,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me4 = ((batch_Y-np.squeeze(temp_result4))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.0          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result5,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me5 = ((batch_Y-np.squeeze(temp_result5))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.1          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result6,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me6 = ((batch_Y-np.squeeze(temp_result6))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.1          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result7,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me7 = ((batch_Y-np.squeeze(temp_result7))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.0          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result8,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me8 = ((batch_Y-np.squeeze(temp_result8))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.0          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result9,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me9 = ((batch_Y-np.squeeze(temp_result9))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result10,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me10 = ((batch_Y-np.squeeze(temp_result10))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result11,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me11 = ((batch_Y-np.squeeze(temp_result11))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.0          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result12,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me12 = ((batch_Y-np.squeeze(temp_result12))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result13,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me13 = ((batch_Y-np.squeeze(temp_result13))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result14,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me14 = ((batch_Y-np.squeeze(temp_result14))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result15,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me15 = ((batch_Y-np.squeeze(temp_result15))**2).mean(axis=None)
          
          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result16,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me16 = ((batch_Y-np.squeeze(temp_result16))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 1.0
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.1          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result17,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me17 = ((batch_Y-np.squeeze(temp_result17))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.0
          beta_sq_value = beta_sq_valu_keep* 1.1          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result18,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me18 = ((batch_Y-np.squeeze(temp_result18))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.0          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result19,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me19 = ((batch_Y-np.squeeze(temp_result19))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.0          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result20,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me20 = ((batch_Y-np.squeeze(temp_result20))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.1          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result21,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me21 = ((batch_Y-np.squeeze(temp_result21))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.1          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result22,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me22 = ((batch_Y-np.squeeze(temp_result22))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 1.1          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result23,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me23 = ((batch_Y-np.squeeze(temp_result23))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 1.1
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result24,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me24 = ((batch_Y-np.squeeze(temp_result24))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 1.1
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result25,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me25 = ((batch_Y-np.squeeze(temp_result25))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 1.1          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result26,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me26 = ((batch_Y-np.squeeze(temp_result26))**2).mean(axis=None)

          ##
          alpha_p_value = alpha_p_valu_keep* 0.9
          alpha_n_value = alpha_n_valu_keep* 0.9
          beta_sq_value = beta_sq_valu_keep* 0.9          
          feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 1.0, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value} 
          _,temp_result27,_,_,_ = sess.run([train_op, logits, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)
          me27 = ((batch_Y-np.squeeze(temp_result27))**2).mean(axis=None)

          ##
          
          me_mat = [me1, me2, me3, me4, me5, me6, me7, me8, me9, me10, me11, me12, me13, me14, me15, me16, me17, me18, me19, me20, me21, me22, me23, me24, me25, me26, me27]
          if np.argmin(me_mat) == 0:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('1')
          
          elif np.argmin(me_mat) == 1:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('2')
          
          elif np.argmin(me_mat) == 2:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('3')
          
          elif np.argmin(me_mat) == 3:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('4')
          
          elif np.argmin(me_mat) == 4:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('5')
          
          elif np.argmin(me_mat) == 5:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('6')
          
          elif np.argmin(me_mat) == 6:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('7')
          
          elif np.argmin(me_mat) == 7:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('8')
          
          elif np.argmin(me_mat) == 8:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('9')
          
          elif np.argmin(me_mat) == 9:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('10')
          
          elif np.argmin(me_mat) == 10:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('11')
          
          elif np.argmin(me_mat) == 11:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('12')
          
          elif np.argmin(me_mat) == 12:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('13')
          
          elif np.argmin(me_mat) == 13:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('14')
          
          elif np.argmin(me_mat) == 14:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('15')
          
          elif np.argmin(me_mat) == 15:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('16')
          
          elif np.argmin(me_mat) == 16:
             alpha_p_value = alpha_p_valu_keep* 1.0
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('17')
          
          elif np.argmin(me_mat) == 17:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.0
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('18')
          
          elif np.argmin(me_mat) == 18:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('19')
          
          elif np.argmin(me_mat) == 19:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.0
             print('20')
          
          elif np.argmin(me_mat) == 20:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('21')
          
          elif np.argmin(me_mat) == 21:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('22')
          
          elif np.argmin(me_mat) == 22:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('23')
          
          elif np.argmin(me_mat) == 23:
             alpha_p_value = alpha_p_valu_keep* 1.1
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('24')
          
          elif np.argmin(me_mat) == 24:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 1.1
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('25')
          
          elif np.argmin(me_mat) == 25:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 1.1
             print('26')
          
          else:
             alpha_p_value = alpha_p_valu_keep* 0.9
             alpha_n_value = alpha_n_valu_keep* 0.9
             beta_sq_value = beta_sq_valu_keep* 0.9
             print('27')
          

      feed_dict = {images_placeholder: batch_X, labels_placeholder: batch_Y, keep_prob: 0.6, phase_train: True, alpha_p: alpha_p_value, alpha_n: alpha_n_value, beta_sq: beta_sq_value}
      
      _, loss_value, alpha_p_value, alpha_n_value, beta_sq_value = sess.run([train_op, loss, updated_alpha_p, updated_alpha_n, updated_beta_sq], feed_dict=feed_dict)    
    
    
      duration = time.time() - start_time    
    

      if (step + 1) % 1 == 0:
         checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
         saver.save(sess, checkpoint_file, global_step=step)
         print('%%%% save the model paramters ... ')
         print('Step %d: loss = %.7f (%.3f sec)' % (step, loss_value, duration))
        
      if (step + 1) % 1 == 0:
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
