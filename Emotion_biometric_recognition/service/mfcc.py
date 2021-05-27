import librosa
import numpy as np
from tensorflow.python.ops import gen_audio_ops as audio_ops
from matplotlib.pyplot import specgram
from tensorflow.python.ops import io_ops
import tensorflow as tf
import soundfile as sf


def cut_data(data, length):
    if data.shape[0] > length:
        data = data[:length]
    elif data.shape[0] < length:
        need_len = length - data.shape[0]
        add_len = need_len // 2
        data = np.concatenate([add_len * [0], data, (need_len - add_len) * [0]])
    return data 


def get_mfcc(path, flags):
    data, sr = librosa.load(path, flags['sample_rate'])
    data = cut_data(data, flags['audio_len'])
    wav_path_for_convert = 'temp_for_convert.wav'
    sf.write(wav_path_for_convert, data, sr)

    wav_loader = io_ops.read_file(wav_path_for_convert)
    waveform = audio_ops.decode_wav(wav_loader, desired_channels=1)    
    spectrogram = audio_ops.audio_spectrogram(
        waveform.audio, 
        window_size=int((sr * flags['window_size']) / 1000), 
        stride=int(sr * flags['stride'] / 1000)
    )
    spectrogram = tf.cast(spectrogram, float)
    mfcc = audio_ops.mfcc(
        spectrogram=spectrogram,
        sample_rate=sr,
        upper_frequency_limit=flags['upper_frequency_limit'],
        lower_frequency_limit=flags['lower_frequency_limit'],
        filterbank_channel_count=flags['filterbank_channel_count'],
        dct_coefficient_count=flags['dct_coefficient_count']
    )
    mfcc = tf.squeeze(mfcc, axis=0)
    mfcc = mfcc.numpy()
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc


def make_mfcc_data(df, flags):
    X = []
    for index, path in enumerate(df.path):  
        mfcc = get_mfcc(path, flags)
        X.append(mfcc)
        if flags['debug'] and index % 100 == 0:
            print(f'Current index = {index}')
    X = np.concatenate(X, axis=0)
    return X
