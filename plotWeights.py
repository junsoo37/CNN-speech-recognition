# code written by Noh Hyun-kyu POSTECH Oct 2019
from __future__ import absolute_import, division, print_function, unicode_literals

import tkinter, pyaudio
from playsound import playsound
import tensorflow as tf
import wave, os, numpy, scipy, librosa, cnn_model
import matplotlib.pyplot as plt
from scipy.io import wavfile

cnn_classifier = tf.estimator.Estimator(
    model_fn=cnn_model.cnn_model, model_dir="./weight_bias_dir")

print('variable names:\n', cnn_classifier.get_variable_names())
# ['conv1d/bias', 'conv1d/kernel', 'dense/bias', 'dense/kernel', 'global_step']
weight_cnn1d = numpy.array(cnn_classifier.get_variable_value('conv1d/kernel'))
weight_dense = numpy.array(cnn_classifier.get_variable_value('dense/kernel'))
print('weight of DNN\n',weight_dense)
conv1d_bias = numpy.array(cnn_classifier.get_variable_value('conv1d/bias'))
print('bias of conv1d\n',conv1d_bias)
dense_bias = numpy.array(cnn_classifier.get_variable_value('dense/bias'))
print('bias of DNN\n',dense_bias)
print('======================================')
print('shape of weight_cnn1d = ',weight_cnn1d.shape)
print('shape of cnn1d_bias = ',conv1d_bias.shape)
print(weight_cnn1d[:,:,0])
print(conv1d_bias[0])
print('======================================')

plt.figure(1)
plt.subplot(3,2,1)
arr= weight_cnn1d[:,:,0]
arr= numpy.transpose(arr)
print(arr.shape)
plt.contour(arr)
plt.title("trained filter weights of conv1d")
plt.ylabel("mel frequency")
plt.subplot(3,2,2)
arr= weight_cnn1d[:,:,1]
arr= numpy.transpose(arr)
plt.contour(arr)
plt.subplot(3,2,3)
arr= weight_cnn1d[:,:,2]
arr= numpy.transpose(arr)
plt.contour(arr)
plt.subplot(3,2,4)
arr= weight_cnn1d[:,:,3]
arr= numpy.transpose(arr)
plt.contour(arr)
plt.subplot(3,2,5)
arr= weight_cnn1d[:,:,4]
arr= numpy.transpose(arr)
plt.contour(arr)
plt.xlabel("normalized frame number")
plt.show()

plt.figure(2)
plt.contourf(weight_dense)
plt.title("trained filter weights of dnn")
plt.show()
##########################################################################
