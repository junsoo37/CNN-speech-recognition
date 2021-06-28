## code written by Noh Hyun-kyu POSTECH Oct 2019
from __future__ import absolute_import, division, print_function, unicode_literals
import tkinter, pyaudio
import tensorflow.compat.v1 as tf
import wave, os, numpy, scipy, librosa, cnn_model
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import play
import normalize_in_time


def estimate_speech_command():
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    frames = []  # Initialize array to store frames
    # Store data in chunks for 2 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk) 	# reda chunk items and store into data
        frames.append(data)
    stream.stop_stream()
    stream.close()
    t = b''.join(frames)
    waveform = numpy.fromstring(t, numpy.int16)
    #print('type and length of frames = ',type(frames), len(frames))
    # play audio sound
    filename="test.wav"
    if os.path.isfile(filename):
        os.remove(filename)
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    play(AudioSegment.from_wav(filename))
    waveform_normalized_in_time = normalize_in_time.normalize_array(waveform)
    S = librosa.feature.melspectrogram(y=waveform_normalized_in_time, n_mels=40, n_fft=input_nfft, hop_length=input_stride, window=scipy.signal.windows.hann) 
    #S = librosa.feature.melspectrogram(y=waveform_normalized_in_time[0 : input_stride * 196], n_mels=40, n_fft=input_nfft, hop_length=input_stride, window=scipy.signal.windows.hann) 
    S_log = numpy.log10(S + 1e-5)
    S_log_100 = normalize_in_time.normalize_in_time(S_log,waveform_normalized_in_time,S,1)   # 1 : record
    #S_log_100 = normalize_in_time.normalize_in_time(S_log,waveform_normalized_in_time,S,3)   # 3 : INFERENCE
    T = []
    T.append(S_log_100)
    T = numpy.array(T)
    input_for_estimation = tf.estimator.inputs.numpy_input_fn(x={"x": T}, y=None, batch_size=1, num_epochs=1, shuffle=False)
    estimated_result = train_eval_cnn_model.predict(input_fn=input_for_estimation)
    pred_dict = next(estimated_result)
    #print('variable names:\n', train_eval_cnn_model.get_variable_names())
    #variable names:
    # ['conv1d/bias', 'conv1d/kernel',              # conv1
    # ['conv1d_1/bias', 'conv1d_1/kernel',          # conv2
    # 'dense/bias', 'dense/kernel', 'global_step']  # dnn
    #print('='*50)
#    kernel_cnn1d = train_eval_cnn_model.get_variable_value('conv1d_1/kernel')
#    bias_cnn1d = train_eval_cnn_model.get_variable_value('conv1d_1/bias')
#    weight_conv1d = numpy.array(kernel_cnn1d)
#    bias_conv1d = numpy.array(bias_cnn1d)
#    out_conv1d=numpy.array(pred_dict['conv2output'])

#    axe3=plt.subplot(223)
#    array2 = numpy.array(pred_dict['conv1output']);
#    axe3.plot(numpy.reshape(array2,array2.shape[1]),'bo')

    pred_dict_probability =pred_dict['probabilities']
    pred_dict_val =pred_dict['classes']
    if(pred_dict_val==0):
        label.config(text='ALEXA ('+str(round(pred_dict_probability[0]*100))+' %)')
    elif(pred_dict_val==1):
        label.config(text='BIXBY ('+str(round(pred_dict_probability[1]*100))+' %)')
    elif(pred_dict_val==2):
        label.config(text='GOOGLE ('+str(round(pred_dict_probability[2]*100))+' %)')
    elif(pred_dict_val==3):
        label.config(text='JINIYA ('+str(round(pred_dict_probability[3]*100))+' %)')
    elif(pred_dict_val==4):
        label.config(text='KLOVA ('+str(round(pred_dict_probability[4]*100))+' %)')
    else:
        print('Error: no such pred_dict value')
    label.pack()
    #print('probabilities: ALEXA:', pred_dict_probability[0],'\t BIXBY:',pred_dict_probability[1],'\t GOOGLE:',pred_dict_probability[2],'\t JINIYA:',pred_dict_probability[3],'\t KLOVA:',pred_dict_probability[4])
    plt.show(block=False)
    template = ('probabilities: ALEXA: {:.1f}% \t  BIXBY: {:.1f}% \t  GOOGLE: {:.1f}% \t  JINIYA: {:.1f}% \t  KLOVA: {:.1f}%') 
    print(template.format(pred_dict_probability[0] * 100, pred_dict_probability[1] * 100, pred_dict_probability[2] * 100, pred_dict_probability[3] * 100, pred_dict_probability[4] * 100))

# MAIN program of inference.py
window=tkinter.Tk()
window.title("INFERENCE")
window.geometry("300x150+100+100")
window.resizable(False, False)

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 48000  # 48000 samples per second
seconds = 2
frame_length = 0.020 
frame_stride = 0.010 
input_nfft = int(round(fs*frame_length)) 
input_stride = int(round(fs*frame_stride)) 
p = pyaudio.PyAudio()  # Create an interface to PortAudio
train_eval_cnn_model = tf.estimator.Estimator(model_fn=cnn_model.cnn_model, model_dir="./weight_bias_dir")
label = tkinter.Label(window, text="Test stage")
label.pack()
button_record = tkinter.Button(window, overrelief="solid", text="Record", width=7, height=1, command=estimate_speech_command, repeatdelay=1000, repeatinterval=100)
button_record.place(x = 125,y = 90)
window.mainloop()
p.terminate()
