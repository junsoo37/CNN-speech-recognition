# 193 (16k sample, 2 sec) changed to 100 for 48k sample 2 sec
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
import numpy as np
model = 4 # choose one of 1, 2, 3, 4

#tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 40, 100 ])
  transposed_input = tf.transpose(input_layer ,perm = [0,2,1])
      ############################# MODEL 1 ###############################################
  if model == 1 :
      conv1 = tf.layers.conv1d( inputs=transposed_input, filters=5, kernel_size=100, strides = 1, padding="valid", activation=tf.nn.relu)
      pool5 = tf.reshape(conv1, [-1, 5])
      logits = tf.layers.dense(inputs=pool5, units=5)	# dnn 50*5 weights 5 bias
      ############################# MODEL 2 ###############################################
  elif model == 2 :
      conv1 = tf.layers.conv1d( inputs=transposed_input, filters=80, kernel_size=100, strides = 1, padding="valid", activation=tf.nn.relu)
      pool5 = tf.reshape(conv1, [-1, 80])
      logits = tf.layers.dense(inputs=pool5, units=5)	# dnn 50*5 weights 5 bias
      ############################# MODEL 3 ###############################################
  elif model == 3 :
      conv1 = tf.layers.conv1d( inputs=transposed_input, filters=20, kernel_size=20, strides = 2, padding="same", activation=tf.nn.relu)
      conv2 = tf.layers.conv1d( inputs=conv1, filters=5, kernel_size=50, padding="valid", activation=tf.nn.relu)
      pool5 = tf.reshape(conv2, [-1, 5])
      logits = tf.layers.dense(inputs=pool5, units=5)	# dnn 5*5 weights 5 bias
      ############################# MODEL 4 ###############################################
  elif model == 4 :
      conv1 = tf.layers.conv1d( inputs=transposed_input, filters=30, kernel_size=20, strides=2, padding="same", activation=tf.nn.relu)
      conv2 = tf.layers.conv1d( inputs=conv1, filters=20, kernel_size=10, strides=2, padding="same", activation=tf.nn.relu)
      conv3 = tf.layers.conv1d( inputs=conv2, filters=10, kernel_size=25, padding="valid", activation=tf.nn.relu)
      pool5 = tf.reshape(conv3, [-1, 10])
  else:
      print('model ',model,': NOT supported')
      return
  ############################# END OF MODEL ###############################################

  # Logits Layer
  logits = tf.layers.dense(inputs=pool5, units=5)	# dnn 13*512*3 weights 3 bias

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)