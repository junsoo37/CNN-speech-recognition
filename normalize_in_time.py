# code written by Noh Hyun-kyu POSTECH Oct 2019
from __future__ import absolute_import, division, print_function, unicode_literals

import pyaudio
#from playsound import playsound
#import CNNtraining.cnn_model_fn as cnn_model_fn
#import tensorflow.compat.v1 as tf
import wave, os, numpy, scipy, librosa #, cnn_model_fn
import matplotlib.pyplot as plt
from scipy.io import wavfile

def normalize_array(array):
    if numpy.max(array) > (-1.0)*numpy.min(array):
        max_array = numpy.max(array) 
    else:
        max_array = (-1.0) * numpy.min(array)
    normalized_array = array / numpy.float(max_array)
    return normalized_array

def normalize_in_time(S_log,float_frames,S,plot_mode):
    min_S_log = numpy.min(S_log)
    max_S_log = numpy.max(S_log)
#    plt.figure(1)
#    plt.subplot(221)
#    plt.title('Waveform, Mel-Spectrogram & power of input data') 
#    plt.plot(float_frames)
#    plt.xlim(0,len(float_frames))
#    plt.xlabel('Sample')
#    plt.ylabel('Amplitude')
    S_log =(S_log-min_S_log)/(max_S_log-min_S_log)      # normalize to the range of [0,1]
    S_power=numpy.zeros(len(S[0]))                 # 1D array of 197
    for k in range(0,len(S_power)):  # 197
        sum=0.0
        for j in range(0,len(S)): # 0 to 40
            sum += S[j][k]**2
        S_power[k] = numpy.log10(sum+1e-5)
    min_S_power = numpy.min(S_power)
    max_S_power = numpy.max(S_power)
    num_frames_continued_increase = 0
    num_frames_continued_above_thresh = 0
    num_frames_continued_below_thresh = 0
    num_frame_start_speech = 0
    num_frame_end_speech = len(S_power)
    #start_threshold = min_S_power + 0.30*(max_S_power-min_S_power)
    start_threshold = min_S_power + 0.50*(max_S_power-min_S_power)
    #end_threshold   = min_S_power + 0.05*(max_S_power-min_S_power)
    end_threshold   = min_S_power + 0.30*(max_S_power-min_S_power)
    for j in range(1,len(S_power)):
        if S_power[j] > start_threshold:
           num_frames_continued_above_thresh += 1
        else:
           num_frames_continued_above_thresh = 0
        if S_power[j] > S_power[j-1]:
           num_frames_continued_increase += 1
        else:
           num_frames_continued_increase = 0
           pass
        #if (num_frames_continued_increase > 3 or (S_power[j]-S_power[j-1]) > start_threshold) and S_power[j] > start_threshold and num_frame_start_speech == 0:
        #if num_frames_continued_increase > 0 and S_power[j] > start_threshold and num_frame_start_speech == 0:
        if S_power[j] > start_threshold and num_frame_start_speech == 0:
           #num_frame_start_speech = j - num_frames_continued_increase
           num_frame_start_speech = j - 10 
           if num_frame_start_speech < 0:
               num_frame_start_speech = 0
        else:
           pass
        if num_frame_start_speech > 0 and S_power[j] < end_threshold:
           num_frames_continued_below_thresh += 1
        else:
           num_frames_continued_below_thresh = 0
#        if num_frames_continued_below_thresh > 5 and num_frame_end_speech == len(S_power):
#           num_frame_end_speech = j
        if num_frames_continued_below_thresh > 30 and num_frame_end_speech == len(S_power):
           num_frame_end_speech = j-15
        elif S_power[j] > start_threshold:
           num_frame_end_speech = len(S_power) 
        else:
           pass
    print('EFFECTIVE NUMBER of FRAMES = ',num_frame_end_speech-num_frame_start_speech+1,'[',num_frame_start_speech,',',num_frame_end_speech,']')
    corrected_S_power=numpy.zeros(len(S_power))
    for i in range(len(S_power)):  # 0 to 197
        #corrected_S_power.append(min_S_power)
        corrected_S_power[i] = min_S_power
    for i in range(num_frame_start_speech, num_frame_end_speech):
        corrected_S_power[i-num_frame_start_speech+0] = S_power[i]
    #plt.plot(corrected_S_power)

    S_log_100=numpy.zeros((40,100))
    for j in range(40):
        for k in range(100):
            t_frame = num_frame_start_speech + k * (num_frame_end_speech - num_frame_start_speech) / 99.0;
            if t_frame > 197:
                t_frame=197
            n = int(t_frame)
            #print('normalize_in_time: t_frame, n = ',t_frame,n)
            S_log_100[j][k] =  (n+1-t_frame) * S_log[j][n] + (t_frame - n ) * S_log[j][n+1]

    if(plot_mode == 1): # RECORD voice and SAVE file
        plt.figure(1)
        plt.subplot(221)
        plt.title('Waveform, Mel-Spectrogram & power of input data') 
        plt.plot(float_frames)
        plt.xlim(0,len(float_frames))
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplot(222)
        plt.contour((S_log-min_S_log)/(max_S_log-min_S_log))
        plt.xlabel('frame number')
        plt.ylabel('Mel Spectrogram')
        #print('row, col of S_log=',len(S_log),len(S_log[0]),'   min & max of S_log =',numpy.min(S_log),numpy.max(S_log))
        #print('row, col of S=',len(S),len(S[0]),'   min & max of S =',numpy.min(S),numpy.max(S))
        plt.subplot(223)
        ##    plt.title('frequency sum of Spectrogram') 
        plt.plot(S_power)
        plt.xlim(0,len(S_power))
        plt.xlabel('frame num')
        plt.ylabel('Power')
        plt.plot(corrected_S_power)
        plt.subplot(224)
        plt.contour((S_log_100-min_S_log)/(max_S_log-min_S_log))
        plt.xlim(0,100)
        plt.xlabel('normalized frame num')
        plt.ylabel('Mel Spectrogram normalized in time')
    elif(plot_mode == 2): # TRAINING
        pass
    elif(plot_mode == 3):   # INFERENCE
        plt.figure(1)
        plt.subplot(221)
        plt.title('Waveform, Mel-Spectrogram & power of input data') 
        plt.plot(float_frames)
        plt.xlim(0,len(float_frames))
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplot(222)
        plt.contour(S_log_100)
        plt.xlabel('frame number')
        plt.ylabel('Mel Spectrogram')
        #print('row, col of S_log_100=',len(S_log_100),len(S_log_100[0]),'   min & max of S_log_100 =',numpy.min(S_log_100),numpy.max(S_log_100))
    else:
        pass
    
    return S_log_100
