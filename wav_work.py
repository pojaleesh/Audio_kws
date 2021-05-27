import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import wavfile
import soundfile as sf
import subprocess
import numpy as np
import os


def build():
    src_filename = '1.oga'
    dest_filename = 'temp.wav'
    process = subprocess.run(['ffmpeg', '-i', src_filename, dest_filename])
    

def delete():
    delete_formats = {'oga', 'wav', 'jpg', 'png', 'ova'}
    for src_filename in os.listdir():
        name = src_filename.split('.')
        if len(name) == 2 and (name[1] in delete_formats):
            process = subprocess.run(['rm', src_filename])
 
    
def read():
    samplerate, data = wavfile.read('temp.wav')
    return (samplerate, data)


def get_lentgh(samplerate, data):
    return len(data) / samplerate


def get_average_amplitude(data):
    return np.average(np.abs(data))


def plot_1(samplerate, data, length):
    time = np.linspace(0., length, len(data))
    plt.plot(time, data, label="Left channel")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.savefig('plot1.png', dpi=300, bbox_inches='tight')
    plt.clf()


def plot_2(samplerate, data):
    spec = plt.specgram(data, NFFT=int(samplerate*0.005), Fs=samplerate, cmap=None, pad_to=256, noverlap=int(samplerate*0.0025))
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency")
    plt.savefig('plot2.png', dpi=300, bbox_inches='tight')
    plt.clf()


def solve():
    build()
    samplerate, data = read()
    length = get_lentgh(samplerate, data)
    average_amplitude = get_average_amplitude(data)
    plot_1(samplerate, data, length)
    plot_2(samplerate, data)
    return average_amplitude
