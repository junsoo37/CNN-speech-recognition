# Libraries: # tkinter : interface toolkit,                 # scipy : scientific tools such as numerical analysis, differential equation
#            # numpy : nuerical computation such as array   # librosa music & audio     # pydub: audio file processing
#            # pyaudio : play and record audio
#import tkinter
import tkinter, pyaudio                 
import wave, os, numpy, scipy, librosa  # scipy : scientific tools such as numerical analysis, differential equation 
import matplotlib.pyplot as plot
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import play
import normalize_in_time

my_student_number = '20189999'      # change 00000000 to your 8-digit student number

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def record_audio_to_wav_file(command_num=1):
    global filename
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    waveform_frames = []  # Initialize a list to store waveform_frames
    # store data in chunks for 2 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        waveform_frames.append(data)
    stream.stop_stream()
    stream.close()
    t = b''.join(waveform_frames)				# convert list to text
    waveform = numpy.fromstring(t, numpy.int16)
    waveform_normalized_in_time = normalize_in_time.normalize_array(waveform)
    S = librosa.feature.melspectrogram(y=waveform_normalized_in_time, n_mels=40, n_fft=input_nfft, hop_length=input_stride, window=scipy.signal.windows.hann) 
    S_log = numpy.log10(S + 1e-5)
    S_log_100 = normalize_in_time.normalize_in_time(S_log,waveform_normalized_in_time,S,1)   # 1 : RECORD voice and SAVE file
    # end the PortAudio interface
    if command_num == 1:
        folder = '../dataset_' + my_student_number + '/ALEXA/'
    elif command_num == 2:
        folder = '../dataset_' + my_student_number + '/BIXBY/'
    elif command_num == 3:
        folder = '../dataset_' + my_student_number + '/GOOGLE/'
    elif command_num == 4:
        folder = '../dataset_' + my_student_number + '/JINIYA/'
    elif command_num == 5:
        folder = '../dataset_' + my_student_number + '/KLOVA/'
    else:
        print('command_num:' + str(command_num) + ' not supported')
    print('folder name:' + folder)
    file_observed = len(os.listdir(folder)) 
    filename = folder + 'set' + str(file_observed) + '.wav'
    label.config(text='Just Recorded ' + filename)
    # save the recorded data into a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(waveform_frames))
    wf.close()
    play(AudioSegment.from_wav(filename))
    button_undo['state'] = 'normal'
    plot.show()

def undo():
    print(filename)
    if os.path.isfile(filename):
        os.remove(filename)
    label.config(text='Recent file is deleted' )
    button_undo['state'] = 'disabled'

# MAIN program of record.py #########################################
window=tkinter.Tk()
window.title("Record Speech Command")
window.geometry("500x150+100+100")
window.resizable(False, False)

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample, 2's complement
channels = 1
fs = 48000  #  sampling rate 48000 samples per second
seconds = 2
frame_length = 0.020    # 20ms for SFFT
frame_stride = 0.010    # each frame proceeds by 10ms
input_nfft = int(round(fs*frame_length)) 
input_stride = int(round(fs*frame_stride)) 

p = pyaudio.PyAudio()  # Create an interface to PortAudio

createFolder('../dataset_' + my_student_number)
createFolder('../dataset_' + my_student_number + '/ALEXA')
createFolder('../dataset_' + my_student_number + '/BIXBY')
createFolder('../dataset_' + my_student_number + '/GOOGLE')
createFolder('../dataset_' + my_student_number + '/JINIYA')
createFolder('../dataset_' + my_student_number + '/KLOVA')

label = tkinter.Label(window, text="Recording mic input to wav file for training")
label.pack()

button_set1 = tkinter.Button(window, overrelief="solid", text="ALEXA", width=7, command=lambda: record_audio_to_wav_file(command_num = 1), repeatdelay=1000, repeatinterval=100)
button_set2 = tkinter.Button(window, overrelief="solid", text="BIXBY", width=7, command=lambda: record_audio_to_wav_file(command_num = 2), repeatdelay=1000, repeatinterval=100)
button_set3 = tkinter.Button(window, overrelief="solid", text="GOOGLE", width=7, command=lambda: record_audio_to_wav_file(command_num = 3), repeatdelay=1000, repeatinterval=100)
button_set4 = tkinter.Button(window, overrelief="solid", text="JINIYA", width=7, command=lambda: record_audio_to_wav_file(command_num = 4), repeatdelay=1000, repeatinterval=100)
button_set5 = tkinter.Button(window, overrelief="solid", text="KLOVA", width=7, command=lambda: record_audio_to_wav_file(command_num = 5), repeatdelay=1000, repeatinterval=100)
button_undo = tkinter.Button(window, overrelief="solid", text="undo", width=7, height=1, command=undo, state = "disabled", repeatdelay=1000, repeatinterval=100)

button_set1.place(x = 10,y = 100)
button_set2.place(x = 110,y = 100)
button_set3.place(x = 210,y = 100)
button_set4.place(x = 310,y = 100)
button_set5.place(x = 410,y = 100)
button_undo.place(x = 210,y = 50)

window.mainloop()
p.terminate()
