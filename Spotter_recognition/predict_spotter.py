import librosa
import librosa.display
import numpy as np
import pandas as pd
import os
import pickle
import soundfile as sf
from pydub import AudioSegment
from Spotter_recognition.mfcc_predict import make_mfcc_prediction
from Spotter_recognition.service.mfcc import get_mfcc_lb, get_mfcc_tf
from Spotter_recognition.service.data_augmentation import normalize_data 

def insert_one(indicator, index):
    for i in range(index, min(indicator.shape[1], index + 851)):
        indicator[0, i] = 1
    return indicator


def make_answer(flags, res):
    cur_start = -1
    cur_end = -1
    cur_word = ''
    ans = ''
    duration = []
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
                    ans = ans + f"You've started to say word {cur_word} at {cur_start / 16000}s.\n"
                    duration.append((cur_word,))
                else:
                    ans = ans + f"You've said word {cur_word} from {cur_start / 16000}s. to {cur_end / 16000}s.\n"
                cur_start = begin_frame
                cur_end = begin_frame + flags['frame_lenght']
                cur_word = word_name
    if cur_start != -1:
        if cur_start == cur_end:
            ans = ans + f"You've started to say word {cur_word} at {cur_start / 16000}s.\n"
        else:
            ans = ans + f"You've said word {cur_word} from {cur_start / 16000}s. to {cur_end / 16000}s.\n"
    return ans


def cut_data(data, sr):
    if len(data) > sr:
        data = data[:sr]
    elif len(data) < sr:
        need_len = sr - len(data)
        add_len = need_len // 2
        data = np.concatenate([add_len * [0], data, (need_len - add_len) * [0]])
    return data


def prepare_data(data, sr):
    data = cut_data(data, sr)
    data = normalize_data(data, sr) 
    save_name = 'interval.wav'
    sf.write(save_name, data, sr)
    return save_name


def make_predict_spotter(model, flags, threshold):
    path = flags['path']
    data, sr = librosa.load(path, sr=flags['sr'])
    indicator = np.zeros((1, len(data)))
    res = []
    predictions = []
    if len(data) <= flags['frame_lenght']:
        save_name = prepate_data(data, flags)
        if flags['mfcc_type'] == 'librosa':
            mfcc = get_mfcc_lb(save_name, sr, flags['mfcc'])
        elif flags['mfcc_type'] == 'tensorflow':
            mfcc = get_mfcc_tf(save_name, sr, flags['mfcc'])
        prediction_1, name_1, prediction_2, name_2 = make_mfcc_prediction(model, flags, mfcc)
        if name_1 == 'unknown':
            predictions.append((prediction_2, name_2, 0))
        else:
            predictions.append((prediction_1, name_1, 0))
        if name_1 != 'unknown' and prediction_1 > threshold:
            insert_one(indicator, 0)
            res.append((name_1, 0, flags['frame_lenght']))
    else:
        begin = 0
        shift = flags['shift']
        while (begin + flags['frame_lenght'] < len(data)):
            save_name = prepare_data(data[begin: begin + flags['frame_lenght']], sr)
            if flags['mfcc_type'] == 'librosa':
                mfcc = get_mfcc_lb(save_name, sr, flags['mfcc'])
            elif flags['mfcc_type'] == 'tensorflow':
                mfcc = get_mfcc_tf(save_name, sr, flags['mfcc'])
            prediction_1, name_1, prediction_2, name_2 = make_mfcc_prediction(model, flags, mfcc)
            if name_1 == 'unknown':
                predictions.append((prediction_2, name_2, begin))
            else:
                predictions.append((prediction_1, name_1, begin))
            if name_1 != 'unknown' and prediction_1 > threshold:
                indicator = insert_one(indicator, begin)
                res.append((name_1, begin, begin + flags['frame_lenght']))
            begin += flags['shift']
    res = make_answer(flags, res)
    return indicator, predictions, res
