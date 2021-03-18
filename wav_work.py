import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import wavfile
import soundfile as sf
import subprocess

def build():
    src_filename = '1.oga'
    dest_filename = 'temp.ogg'
    process = subprocess.call(['ffmpeg', '-i', src_filename, dest_filename], shell=True)
    src_filename = 'temp.ogg'
    dest_filename = 'temp.wav'
    process = subprocess.call(['ffmpeg', '-i', src_filename, dest_filename], shell=True)
    

def delete():
    src_filename = 'temp.wav'
    process = subprocess.call(['del', src_filename], shell=True)
    src_filename = 'temp.ogg'
    process = subprocess.call(['del', src_filename], shell=True)

    
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
    plt.legend()
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