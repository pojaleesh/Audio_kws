import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
import pickle
import IPython.display as ipd
import random
from pydub import AudioSegment
import pathlib

SPOTTER_DATA = "../../spotter_data/"
BACKGROUND_DATA = "../../spotter_data/_background_noise_/"

df_target = pd.read_csv('../../Target_words_dataframe')
df_unknown = pd.read_csv('../../Unknown_words_dataframe')
df_target = df_target.drop(columns=['Unnamed: 0'], axis=1)
df_unknown = df_unknown.drop(columns=['Unnamed: 0'], axis=1)
background_dir = os.listdir(BACKGROUND_DATA)


def pick_random_background(background_dir):
    return random.choice(background_dir)


def pick_random_target_word():
    return random.choice(range(0, df_target.shape[0]))


def pick_random_unknown_word():
    return random.choice(range(0, df_unknown.shape[0]))


def get_background_segment(audio_clip, noise_len):
    start_ms = 0
    while True:
        ind = random.choice(range(0, len(audio_clip)))
        if ind + noise_len < len(audio_clip):
            return audio_clip[ind: ind + noise_len]
        

def get_random_time_segment(background_len, audio_clip_len):
    start_segment = random.choice(range(0, background_len - audio_clip_len))
    return (start_segment, start_segment + audio_clip_len - 1)


def is_intersect(segments, new_segment):
    is_overlapping = False
    segment_start, segment_end = new_segment
    for previous_start, previous_end in segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            is_overlapping = True
    return is_overlapping


def graph_spectogram(path):
    nfft = 200 
    fs = 8000 
    noverlap = 120 
    data, sample_rate = librosa.load(path, sr=44100)
    pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    return pxx;


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def make_sample(flags):
    noise_len = flags['noise_len']
    noise = pick_random_background(background_dir)
    data = AudioSegment.from_wav(BACKGROUND_DATA + noise)
    data = data - 20
    X = get_background_segment(data, noise_len)
    y = np.zeros((1, flags['Ty']))
    segments = []
    cnt = 0
    while (cnt < flags['target_word_count']):
        target_word_index = pick_random_target_word()
        target_word_path = df_target['path'][target_word_index]
        target_word_data = AudioSegment.from_wav(target_word_path)
        target_word_segment = get_random_time_segment(len(X), len(target_word_data))
        start_x = target_word_segment[0]
        end_x = target_word_segment[1]
        end_y = int(end_x * flags['Ty'] / float(noise_len))
        if is_intersect(segments, target_word_segment) == False:
            cnt += 1
            segments.append(target_word_segment)
            for j in range(end_y + 1, end_y + 51):
                if j < flags['Ty']:
                    y[0, j] = 1
            X = X.overlay(target_word_data, position = start_x)
    cnt = 0
    while (cnt < flags['unknown_word_count']):
        unknown_word_index = pick_random_unknown_word()
        unknown_word_path = df_unknown['path'][unknown_word_index]
        unknown_word_data = AudioSegment.from_wav(unknown_word_path)
        unknown_word_segment = get_random_time_segment(len(X), len(unknown_word_data))
        start_x = unknown_word_segment[0]
        end_x = unknown_word_segment[1]
        end_y = int(end_x * flags['Ty'] / float(noise_len))
        if is_intersect(segments, unknown_word_segment) == False:
            cnt += 1
            segments.append(unknown_word_segment)            
            for j in range(end_y + 1, end_y + 51):
                if j < flags['Ty']:
                    y[0, j] = 1
            X = X.overlay(unknown_word_data, position = start_x)
    X = match_target_amplitude(X, -20.0)
    X.export(flags['save_name'], format="wav")
    if flags['data_type'] == "spectogram":
        X = graph_spectogram(save_name)
        return X, y
    elif flags['data_type'] == "mfcc":
        X, sample_rate = librosa.load(
            flags['save_name'], 
            res_type='kaiser_fast',
            duration=flags['duration'],
            sr=flags['sample_rate'],
            offset=flags['offset']
        )  
        mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=flags['n_mfcc'])
        return mfcc, y
    else:        
        raise TypeError("Unknown audio type")
