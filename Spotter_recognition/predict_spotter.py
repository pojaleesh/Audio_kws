import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
import pickle
import IPython.display as ipd
import random
from pydub import AudioSegment
from Spotter_recognition.mfcc_predict import make_mfcc_prediction
from Spotter_recognition.service.Mfcc import get_mfcc

def insert_one(indicator, index):
    for i in range(index, min(indicator.shape[1], index + 51)):
        indicator[0, i] = 1
    return indicator


def make_answer(flags, res):
    cur_start = -1
    cur_end = -1
    cur_word = ''
    ans = ''
    for index in range(len(res)):
        word_name, begin_frame, end_frame = res[index]
        if cur_start == -1:
            cur_start = begin_frame
            cur_end = cur_start
            cur_word = word_name
        else:
            if word_name == cur_word:
                cur_end = begin_frame
            else:
                if cur_start == cur_end:
                    ans = ans + f"You've started to say word {cur_word} at {cur_start / 1000}s.\n"
                else:
                    ans = ans + f"You've said word {cur_word} from {cur_start / 1000}s. to {cur_end / 1000}s.\n"
                cur_start = begin_frame
                cur_end = begin_frame + flags['frame_lenght']
                cur_word = word_name
    if cur_start != -1:
        if cur_start == cur_end:
            ans = ans + f"You've started to say word {cur_word} at {cur_start / 1000}s.\n"
        else:
            ans = ans + f"You've said word {cur_word} from {cur_start / 1000}s. to {cur_end / 1000}s.\n"
    return ans


def make_predict_spotter(model, flags, threshold):
    path = flags['path']
    data = AudioSegment.from_wav(path)
    indicator = np.zeros((1, len(data)))
    print(len(data))
    res = []
    temp = []
    if len(data) <= flags['frame_lenght']:
        mfcc = get_mfcc(path, flags['mfcc'])
        prediction, prediction_name = mfcc_predict(flags, mfcc)
        if prediction_name != 'unknown' and prediction > threshold:
            insert_one(indicator, index)
    else:
        begin = 0
        shift = flags['shift']
        cnt = 0
        while (begin + flags['frame_lenght'] < len(data)):
            cnt += 1
            interval = data[begin: begin + flags['frame_lenght']]
            save_name = 'interval.wav'
            save_path = interval.export(save_name, format="wav")
            mfcc = get_mfcc(save_name, flags['mfcc'])
            prediction, prediction_name = make_mfcc_prediction(model, flags, mfcc)
            print(prediction, prediction_name)
            temp.append(prediction)
            if prediction_name != 'unknown' and prediction > threshold:
                indicator = insert_one(indicator, begin)
                res.append((prediction_name, begin, begin + flags['frame_lenght']))
            begin += flags['shift']
    answer = make_answer(flags, res)
    return indicator, temp, answer
