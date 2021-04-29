import librosa
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import pickle
from pydub import AudioSegment
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, 
    Shift, ClippingDistortion, SpecFrequencyMask
)
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils, to_categorical


def load_background(background_folder):
    background_noises = os.listdir(background_folder)
    noise_to_data = {}
    for noise in background_noises:
        data, sr = librosa.load(background_folder + '/' + noise)
        noise_to_data[noise] = data
    return noise_to_data


def pick_background_folder(background_folder):
    background_noises = os.listdir(background_folder)
    return random.choice(background_noises)


def get_background_segment(background_data, noise_len):
    start_ms = 0
    while True:
        ind = random.choice(range(0, len(background_data)))
        if ind + noise_len < len(background_data):
            return background_data[ind: ind + noise_len] 


def insert_background(background_folder, noise_to_data, wav_data, len_data):
    background_name = pick_background_folder(background_folder)
    background_data = noise_to_data[background_name]
    noise = get_background_segment(background_data, len_data)
    if background_name == 'pink_noise.wav' or background_name == 'white_noise.wav':
        wav_data = 10 * wav_data + noise
    else:
        wav_data = 4 * wav_data + noise
    return wav_data


def make_aug(wav_data, sr, background_folder, noise_to_data, len_data, p_aug):
    augment_gaussian_noise = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p_aug['gaussian'])])
    augment_for_word = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.5, p=p_aug['stretch']),
        PitchShift(min_semitones=-6, max_semitones=6, p=p_aug['pitch_shift']),
        Shift(min_fraction=-0.3, max_fraction=0.3, p=p_aug['shift']),
        ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=15, p=p_aug['clipping'])
    ])
    wav_data = augment_for_word(samples=wav_data, sample_rate=sr)
    background_pick = random.randint(0, 2)
    if background_pick == 0:
        # just silence
        return wav_data
    elif background_pick == 1:
        # background from dataset
        return insert_background(background_folder, noise_to_data, wav_data, len_data)
    else:
        # background is a Gaussian noise
        return augment_gaussian_noise(samples=wav_data, sample_rate=sr)


def batch_to_mfcc(data, sr):
    # TODO refactor this function with code from Mfcc.py
    batch_mfcc = []
    for word_index in range(0, data.shape[0]):
        mfcc = librosa.feature.mfcc(y=data[word_index], sr=sr, n_mfcc=40)
        if mfcc.shape[1] < 61:
            b = np.zeros((40, 61 - mfcc.shape[1]))
            mfcc = np.concatenate((mfcc, b), axis=1)
        if mfcc.shape[1] > 61:
            mfccs = mfccs[:, :61]
        mfcc = np.expand_dims(mfcc, axis=0)
        batch_mfcc.append(mfcc)
    batch_mfcc = np.concatenate(batch_mfcc, axis=0)
    batch_mfcc = np.expand_dims(batch_mfcc, axis=3)
    return batch_mfcc


def make_batch(df, lb, background_folder, noise_to_data, batch_size, len_data, p_aug):
    # batch size must be divisible by 16
    words = df.sample(n=batch_size)
    X = []
    for batch_num in range(0, batch_size // 16):
        batch_df = df.sample(n=16)
        temp = []
        for word_path in batch_df['path']:
            data, sr = librosa.load(word_path)
            data = np.expand_dims(data, 0)
            if len(data) > len_data:
                data = data[:, :len_data]
            elif len(data) < len_data:
                b = np.zeros((1, len_data - data.shape[1]))
                data = np.concatenate((data, b), axis=1)
            data = make_aug(data, sr, background_folder, noise_to_data, len_data, p_aug)
            X.append(data)
    X = np.concatenate(X, axis=0)
    Y = np.array(words['word'])
    Y = np_utils.to_categorical(lb.transform(Y))
    return batch_to_mfcc(X, sr), Y
