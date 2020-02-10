
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math

import tensorflow as tf
import numpy as np

initializer = tf.contrib.layers.xavier_initializer_conv2d()


def deepnn(images, image_size, batch_size, keep_prob, phase_train):
  """Build the mU-Net for prediction
  """

  
  resol = image_size
  
  X_image = tf.reshape(images, [-1, resol, resol, 1])
  
  h1_out1, h1_out2 = CONV(X_image, [3,3,1,32], int(32), int(resol), phase_train)
  h1_out11, h1_out22 = CONV(h1_out1, [3,3,32,64], int(64), int(resol), phase_train)
  h1_out22 = tf.nn.dropout(h1_out22,keep_prob)
  
  h1_out22_ew = UPCONV2(h1_out22, [3,3,64,64], [batch_size, int(resol), int(resol), 64] )

 
  ############################
  
  h2_out1, h2_out2 = CONV(h1_out22, [3,3,64,64], int(64), int(resol/2), phase_train)
  h2_out11, h2_out22 = CONV(h2_out1, [3,3,64,96], int(96), int(resol/2), phase_train)
  h2_out22 = tf.nn.dropout(h2_out22,keep_prob)
  
  h2_out22_ew = UPCONV2(h2_out22, [3,3,96,96], [batch_size, int(resol/2), int(resol/2), 96] )
 
  ############################
  
  h3_out1, h3_out2 = CONV(h2_out22, [3,3,96,96], int(96), int(resol/4), phase_train)
  h3_out11, h3_out22 = CONV(h3_out1, [3,3,96,128], int(128), int(resol/4), phase_train)
  h3_out22 = tf.nn.dropout(h3_out22,keep_prob)
  
  h3_out22_ew = UPCONV2(h3_out22, [3,3,128,128], [batch_size, int(resol/4), int(resol/4), 128] )

  ############################
  
  h4_out1, h4_out2 = CONV(h3_out22, [3,3,128,128], int(128), int(resol/8), phase_train)
  h4_out11, h4_out22 = CONV(h4_out1, [3,3,128,192], int(192), int(resol/8), phase_train)
  h4_out22 = tf.nn.dropout(h4_out22,keep_prob)
  
  h4_out22_ew = UPCONV2(h4_out22, [3,3,192,192], [batch_size, int(resol/8), int(resol/8), 192] )

  ############################
  h5_out1, h5_out2 = CONV(h4_out22, [3,3,192,192], int(192), int(resol/16), phase_train)
  #  h5_out1 = tf.nn.dropout(h5_out1,keep_prob)
  h5_out11, h5_out22 = CONV(h5_out1, [3,3,192,192], int(192), int(resol/16), phase_train)
  h5_out22 = tf.nn.dropout(h5_out22,keep_prob) 
  
  h5_out22_ew = UPCONV2(h5_out22, [3,3,192,192], [batch_size, int(resol/16), int(resol/16), 192] )
  
  ############################
  h6_out = CONV2(h5_out22, [3,3,192,192], int(192), int(resol/32), phase_train)

  h6_out2 = CONV2(h6_out, [3,3,192,192], int(192), int(resol/32), phase_train)
  #####################
  #####################
  #####################
  
  h6_out = UPCONV2(h6_out2, [3,3,192,192], [batch_size, int(resol/16), int(resol/16), 192])
  
  h5_out11 = tf.subtract(h5_out11,prelu(h5_out22_ew))
  h_f5 = tf.image.resize_images(h5_out11,(int(resol),int(resol))) 

  h5_out11 = CONV2(h5_out11,[3,3,192,192], int(192), int(resol/16), phase_train)
  
  h6_out = tf.concat([h6_out,h5_out11], 3)
  u0_out = CONV2(h6_out, [3,3,384,192],int(192), int(resol/16), phase_train)
  u0_out = tf.nn.dropout(u0_out,keep_prob) 
  
 
  u0_out = UPCONV2(u0_out, [3,3,128,192], [batch_size, int(resol/8), int(resol/8), 128])
  
  h4_out11 = tf.subtract(h4_out11,prelu(h4_out22_ew))
  
  h_f4 = tf.image.resize_images(h4_out11,(int(resol),int(resol))) 
  
  h4_out11 = CONV2(h4_out11,[3,3,192,192], int(192), int(resol/8), phase_train)
  
  u0_out = tf.concat([u0_out,h4_out11], 3)
  u1_out = CONV2(u0_out, [3,3,320,128],int(128), int(resol/8), phase_train)
  u1_out = tf.nn.dropout(u1_out,keep_prob)  

 
  u1_out = UPCONV2(u1_out, [3,3,96,128], [batch_size, int(resol/4), int(resol/4), 96])
  
  h3_out11 = tf.subtract(h3_out11,prelu(h3_out22_ew))
  
  h_f3 = tf.image.resize_images(h3_out11,(int(resol),int(resol))) 
  h3_out11 = CONV2(h3_out11,[3,3,128,128], int(128), int(resol/4), phase_train)
  
  u1_out = tf.concat([u1_out, h3_out11], 3) 
  u2_out = CONV2(u1_out, [3,3,224,96], int(96), int(resol/4), phase_train)
  u2_out = tf.nn.dropout(u2_out,keep_prob)  
  
 
  u2_out = UPCONV2(u2_out, [3,3,64,96], [batch_size, int(resol/2), int(resol/2), 64])
  
  
  h2_out11 = tf.subtract(h2_out11,prelu(h2_out22_ew))
  
  h_f2 = tf.image.resize_images(h2_out11,(int(resol),int(resol))) 
  h2_out11 = CONV2(h2_out11,[3,3,96,96], int(96), int(resol/2), phase_train)
  u2_out = tf.concat([u2_out,h2_out11], 3) 
  u3_out = CONV2(u2_out, [3,3,160,64], int(64), int(resol/2), phase_train)
  u3_out = tf.nn.dropout(u3_out,keep_prob) 
  
 
  u3_out = UPCONV2(u3_out, [3,3,32,64], [batch_size, int(resol/1), int(resol/1), 32])
  
  h1_out11 = tf.subtract(h1_out11,prelu(h1_out22_ew))
  
  h_f1 = tf.image.resize_images(h1_out11,(int(resol),int(resol))) 
  h1_out11 = CONV2(h1_out11,[3,3,64,64], int(64), int(resol/1), phase_train)
     
  u3_out = tf.concat([u3_out, h1_out11], 3) 
  u4_out = CONV2(u3_out, [3,3,96,32], int(32), int(resol/1), phase_train)
  u4_out = tf.nn.dropout(u4_out,keep_prob)  
  
  u4_out = FCONV(u4_out, [1,1,32,1])   
  y_out = tf.add(u4_out,X_image)
  
  
  return y_out


       
    
def GLF(logits, labels, alpha_p, alpha_n, beta_sq):
    
    smooth = 1e-10
    y_true = tf.keras.backend.flatten(labels)
    y_pred = tf.keras.backend.flatten(logits)
    
    y_tp = tf.keras.backend.sum(y_true * y_pred * y_true * y_pred)
    y_fp = tf.keras.backend.sum(y_pred * (1 - y_true) * y_pred * (1 - y_true))
    y_fn = tf.keras.backend.sum((1 - y_pred) * y_true * (1 - y_pred) * y_true)
    
    p = (y_fp + smooth) / (y_tp + y_fp + smooth)
    r = (y_fn + smooth) / (y_tp + y_fn + smooth)
    
   
    
    c2 = tf.logical_and(y_fp/2 < y_tp, y_tp < 2*y_fp)
    c2 = tf.cast(c2,dtype=tf.float32)
    c7 = tf.logical_and(y_fp/15 < y_tp, y_tp < y_fp/2)
    c7 = tf.cast(c7,dtype=tf.float32)
    c8 = y_tp < y_fp/15
    c8 = tf.cast(c8,dtype=tf.float32)
    c11 = tf.logical_and(2*y_fp < y_tp, y_tp < 15*y_fp)
    c11 = tf.cast(c11,dtype=tf.float32)
    c12 = tf.less(15*y_fp, y_tp)
    c12 = tf.cast(c12,dtype=tf.float32)
    
        
    up_ap = 1.0*c2 + 0.75*c7 + 0.5*c8 + 1.25*c11 + 1.5*c12 
    
    up_ap2 = tf.cond(y_fp < tf.constant(5.), lambda: 1.0, lambda: up_ap)
      
        
    
    c2 = tf.logical_and(y_fn/2 < y_tp, y_tp < 2*y_fn)
    c2 = tf.cast(c2,dtype=tf.float32)
    c7 = tf.logical_and(y_fn/15 < y_tp, y_tp < y_fn/2)
    c7 = tf.cast(c7,dtype=tf.float32)
    c8 = y_tp < y_fn/15
    c8 = tf.cast(c8,dtype=tf.float32)
    c11 = tf.logical_and(2*y_fn < y_tp, y_tp < 15*y_fn)
    c11 = tf.cast(c11,dtype=tf.float32)
    c12 = tf.less(15*y_fn, y_tp)
    c12 = tf.cast(c12,dtype=tf.float32)
    
    c15 = y_fn < tf.constant(1.)
    c15 = tf.cast(c15,dtype=tf.float32)
    
    up_an = 1.0*c2 + 0.75*c7 + 0.5*c8 + 1.25*c11 + 1.5*c12 
    
    up_an2 = tf.cond(y_fn < tf.constant(1.), lambda: 1.0, lambda: up_an)
       
    
    alpha_p2 = alpha_p*up_ap2
    alpha_n2 = alpha_n*up_an2
    
    
    
    p_ap = (y_tp + smooth) / (y_tp + y_fp * alpha_p2 + smooth)
    r_an = (y_tp + smooth) / (y_tp + y_fn * alpha_n2 + smooth)
    
    
    
    c2 = tf.logical_and(p_ap/2 < r_an, r_an < 2*p_ap)
    c2 = tf.cast(c2,dtype=tf.float32)
    c7 = tf.logical_and(p_ap/15 < r_an, r_an < p_ap/2)
    c7 = tf.cast(c7,dtype=tf.float32)
    c8 = r_an < p_ap/15
    c8 = tf.cast(c8,dtype=tf.float32)
    c11 = tf.logical_and(2*p_ap < r_an, r_an < 15*p_ap)
    c11 = tf.cast(c11,dtype=tf.float32)
    c12 = tf.less(15*p_ap, r_an)
    c12 = tf.cast(c12,dtype=tf.float32)
    
    up_bsq = 1.0*c2 + 0.75*c7 + 0.5*c8 + 1.25*c11 + 1.5*c12 

    
    beta_sq2 = beta_sq*up_bsq
    
      
    
    answer = ( (1+beta_sq2)*y_tp + smooth ) / ( (1+beta_sq2)*y_tp + alpha_p2 * y_fp + beta_sq2 * alpha_n2 * y_fn + smooth )
       
    return answer, alpha_p2, alpha_n2, beta_sq2  


def lossfn(logits, labels, alpha_p, alpha_n, beta_sq):
    answer, alpha_p2, alpha_n2, beta_sq2 = GLF(logits, labels, alpha_p, alpha_n, beta_sq)
    return 1. -  answer, alpha_p2, alpha_n2, beta_sq2  
 

    
def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
  
  
def CONV(INPUT, shape, fill2, scale, phase_train):
  W = tf.Variable(initializer(shape=shape))
  b = tf.Variable(tf.constant(0.1, shape=[shape[3]]))
  h = tf.nn.conv2d(INPUT, W, strides=[1,1,1,1], padding='SAME')
  h_bn = batch_norm(tf.reshape(h, shape = [-1, scale, scale, fill2]), fill2, phase_train)
  h_relu = prelu(h_bn)
  
  h_pool = tf.nn.max_pool(h_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  return h_relu, h_pool
  



def CONV2(INPUT, shape, fill2, scale, phase_train):
  W = tf.Variable(initializer(shape=shape))
  b = tf.Variable(tf.constant(0.1, shape=[shape[3]]))
  h = tf.nn.conv2d(INPUT, W, strides=[1,1,1,1], padding='SAME')
  h_bn = batch_norm(tf.reshape(h, shape = [-1, scale, scale, fill2]), fill2, phase_train)
  h_relu = prelu(h_bn)
  
  return h_relu
  
  
    

  
  
def UPCONV2(INPUT, W_shape, output_shape):
  W = tf.Variable(initializer(shape=W_shape))
  b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
  h = tf.nn.conv2d_transpose(INPUT, W, output_shape=output_shape, strides=[1,2,2,1])

  return h

  
  
def FCONV(INPUT, shape):
  W = tf.Variable(initializer(shape=shape))
  b = tf.Variable(tf.constant(0.1, shape=[shape[3]]))
  h = tf.nn.conv2d(INPUT, W, strides=[1,1,1,1], padding='SAME')
  h_relu = prelu(h)
  return h_relu
  
  
def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
        

  
def ewm2d(x,W):
    return tf.multiply(x, W, name = None)
    

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

    
    


  
  
  
  
  