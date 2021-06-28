# copy this file into your dataset_20180???? directory and run by typing python check_wavfiles.py
import pyaudio, wave, os
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import play
import scipy
import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
import keyboard  
import normalize_in_time  

def normalize_array(array):
    if np.max(array) > (-1.0)*np.min(array):
        max_array = np.max(array) 
    else:
        max_array = (-1.0) * np.min(array)
    normalized_array = array / np.float(max_array)
    return normalized_array

fs = 48000
stride_time = 0.02
hop_time = 0.01
input_stride = int(hop_time * fs)
input_nfft = int(stride_time * fs)

keywords = ['ALEXA', 'BIXBY', 'GOOGLE', 'JINIYA', 'KLOVA']
for keyword in keywords:
   wav_folder = './' + keyword
   print(keyword, 'files count: ', num_files)
   filename_list =os.listdir(wav_folder)

   for filename in filename_list:
      wav_file = wav_folder + '/' + filename
      y, sr = librosa.load(wav_file, sr=fs) 
      y = normalize_array(np.array(y))
      play(AudioSegment.from_wav(wav_file))
      print(keyword,': ',wav_file,': ',end='\t')
   
      S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride, window=scipy.signal.windows.hann) 
      S_log = np.log10(S + 1e-5)
      S_log_100 = normalize_in_time.normalize_in_time(S_log,y,S,1)   # 1 : RECORD voice and SAVE file
      plt.show(block=False)
      plt.pause(1)
      while True :
         try:  
          if keyboard.is_pressed('enter'):  
              plt.close()
              break
          else:
              pass
         except:
          break 
      #plt.close()
