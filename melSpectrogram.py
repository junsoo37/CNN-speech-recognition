# code written by Noh Hyun-kyu POSTECH Oct 2019
# -*- coding: utf-8 -*- 
import random
import librosa 
import librosa.display 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal
import os
import normalize_in_time
import struct
frame_length = 0.020 
frame_stride = 0.010
testset_ratio = 0.2
foldername = '../dataset_students'

def generate_spectrogram(wav_file):
    y, sr = librosa.load(wav_file, sr=48000) 
    #print(keyword,': file_name=',file_name,end="\t") # HJP
    y = normalize_in_time.normalize_array(y)
    input_nfft = int(round(sr*frame_length)) 
    input_stride = int(round(sr*frame_stride)) 
    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride, window=scipy.signal.windows.hann) 
    S_log = np.log10(S + 1e-5)    
    S_log_100 = normalize_in_time.normalize_in_time(S_log,y,S,2) 
    return S_log_100

def WriteTrainingSet(Small_mode = False):
    keywords = ['ALEXA', 'BIXBY', 'GOOGLE', 'JINIYA', 'KLOVA']
    X_training = []
    Y_training = [] 
    X_test = []
    Y_test = []
    labels = 0
    os.system('del ' + foldername + '/' +  'train_data.bin')
    os.system('del ' + foldername + '/' +  'train_label.bin')
    os.system('del ' + foldername + '/' +  'test_data.bin')
    os.system('del ' + foldername + '/' +  'test_label.bin')
    fp_train_data = open(foldername + '/' + 'train_data.bin','wb')
    fp_train_label = open(foldername + '/' + 'train_label.bin','wb')
    fp_test_data = open(foldername + '/' + 'test_data.bin','wb')
    fp_test_label = open(foldername + '/' + 'test_label.bin','wb')
    if Small_mode :
        for keyword in keywords:  
            wav_folder = foldername + '/' + keyword    
            file_observed = os.listdir(wav_folder)
            random.shuffle(file_observed)
            print('converting ',file_observed,' files of ', keyword, ':')
            test_num = int(len(file_observed) * testset_ratio)
            fp_train_data.write(struct.pack('i', len(file_observed) - test_num))
            fp_train_label.write(struct.pack('i' ,len(file_observed) - test_num))
            fp_test_data.write(struct.pack('i', test_num))
            fp_test_label.write(struct.pack('i' ,test_num))

            idx = 0
            for file_name in file_observed:
                wav_file = wav_folder + '/' + file_name
                S_log_100 = generate_spectrogram(wav_file)
                if(idx < test_num) : 
                    fp_test_data.write(struct.pack('d'*100 * 40, *S_log_100.flatten() ))
                    fp_test_label.write(struct.pack('i' , labels))
                else : 
                    fp_train_data.write(struct.pack('d'*100 * 40, *S_log_100.flatten() ))
                    fp_train_label.write(struct.pack('i' , labels))
                idx = idx + 1
            labels = labels + 1
    else :
        for keyword in keywords:   
            wav_folder = foldername + '/train/' + keyword   
            file_observed = os.listdir(wav_folder)
            file_observed.sort()
            print('converting ',file_observed,' files of ', keyword, ':')
            fp_train_data.write(struct.pack('i', len(file_observed)))
            fp_train_label.write(struct.pack('i' ,len(file_observed)))

            for file_name in file_observed:
                wav_file = wav_folder + '/' + file_name
                S_log_100 = generate_spectrogram(wav_file)
                fp_train_data.write(struct.pack('d'*100 * 40, *S_log_100.flatten() ))
                fp_train_label.write(struct.pack('i' , labels))
                title_str = keyword + file_name
                plt.title(title_str)
            labels = labels + 1
        labels = 0
        for keyword in keywords:
            wav_folder = foldername + '/test/' + keyword       
            file_observed = os.listdir(wav_folder)
            file_observed.sort()
            print('converting ',file_observed,' files of ', keyword, ':')
            fp_test_data.write(struct.pack('i', len(file_observed)))
            fp_test_label.write(struct.pack('i' ,len(file_observed)))

            for file_name in file_observed:
                wav_file = wav_folder + '/' + file_name
                S_log_100 = generate_spectrogram(wav_file)
                fp_test_data.write(struct.pack('d'*100*40 , *S_log_100.flatten() ))
                fp_test_label.write(struct.pack('i' , labels))
                title_str = keyword + file_name
                plt.title(title_str)
            labels = labels + 1

    fp_train_label.close()
    fp_test_label.close()
    fp_train_data.close()
    fp_test_data.close()

def ReadTrainingSet():
    fp_train_data = open(foldername + '/' + 'train_data.bin','rb')
    fp_train_label = open(foldername + '/' + 'train_label.bin','rb')
    fp_test_data = open(foldername + '/' + 'test_data.bin','rb')
    fp_test_label = open(foldername + '/' + 'test_label.bin','rb')
    X_training = []
    Y_training = [] 
    X_test = []
    Y_test = []
    keywords = ['ALEXA', 'BIXBY', 'GOOGLE', 'JINIYA', 'KLOVA']
    for keyword in keywords:
        file_observed =   (struct.unpack('i', fp_train_data.read(4)))[0]
        file_observed =   (struct.unpack('i', fp_train_label.read(4)))[0]
        print('Train : ' + str(file_observed))   
        for i in range(file_observed):
            S_log_100 = np.array(struct.unpack('d'* 100 * 40,fp_train_data.read(8*100*40))).reshape((40, 100))
            labels = (struct.unpack('i',fp_train_label.read(4)))[0]
            X_training.append(S_log_100)
            Y_training.append(labels) 

    for keyword in keywords:
        file_observed =   (struct.unpack('i', fp_test_data.read(4)))[0]
        file_observed =   (struct.unpack('i', fp_test_label.read(4)))[0]
        print('Test : ' + str(file_observed))
        for i in range(file_observed):
            S_log_100 = np.array(struct.unpack('d'* 100 * 40,fp_test_data.read(8*100*40))).reshape((40, 100))
            labels = (struct.unpack('i',fp_test_label.read(4)))[0]
            X_test.append(S_log_100)
            Y_test.append(labels) 

    fp_train_label.close()
    fp_test_label.close()
    fp_train_data.close()
    fp_test_data.close()

    return X_training, Y_training, X_test, Y_test 
