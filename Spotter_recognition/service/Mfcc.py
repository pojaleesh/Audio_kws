import librosa
import numpy as np
from tensorflow.python.ops import gen_audio_ops as audio_ops
from matplotlib.pyplot import specgram
from tensorflow.python.ops import io_ops
import tensorflow as tf


def get_mfcc_lb(path, flags):
    X, sample_rate = librosa.load(
        path, res_type='kaiser_fast', 
        duration=flags['duration'], 
        sr=flags['sample_rate'], 
        offset=flags['offset']
    )
    sample_rate = np.array(sample_rate)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=flags['n_mfcc'])
    if mfccs.shape[1] < flags['columns']:
        b = np.zeros((flags['n_mfcc'], flags['columns'] - mfccs.shape[1]))
        mfccs = np.concatenate((mfccs, b), axis=1)
    if mfccs.shape[1] > flags['columns']:
        mfccs = mfccs[:, :flags['columns']]
    mfccs = np.expand_dims(mfccs, axis=0)
    return mfccs


def get_mfcc_tf(path, sr, flags):
    wav_loader = io_ops.read_file(path)
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
        if flags['mfcc_type'] == 'tensroflow':
            mfcc = get_mfcc_tf(path, flags)
        elif flags['mfcc_type'] == 'librosa':
            mfcc = get_mfcc_lb(path, flags)
        X.append(mfcc)
        if flags['debug'] and index % 100 == 0:
            print(f'Current index = {index}')
    X = np.concatenate(X, axis=0)
    return X
