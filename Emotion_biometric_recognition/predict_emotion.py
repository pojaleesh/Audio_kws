import librosa
import librosa.display
import numpy as np
import pandas as pd
import os
import pickle
import soundfile as sf
from collections import defaultdict
from Emotion_biometric_recognition.mfcc_predict import make_mfcc_prediction
from Emotion_biometric_recognition.service.mfcc import get_mfcc


def prepare_data(data, sr):
    save_name = 'interval.wav'
    print(sr)
    sf.write(save_name, data, sr)
    return save_name


def make_predict_emotion(model, flags):
    path = flags['path']
    data, sr = librosa.load(path, sr=flags['sr'])
    res = defaultdict(int) 
    if len(data) <= flags['frame_lenght']:
        save_name = prepare_data(data, sr)
        mfcc = get_mfcc(save_name, flags['mfcc'])
        prediction, prediction_name = make_mfcc_prediction(model, flags, mfcc)
        return prediction_name
    else:
        begin = 0
        shift = flags['shift']
        while (begin + flags['frame_lenght'] < len(data)):
            save_name = prepare_data(data[begin: begin + flags['frame_lenght']], sr)
            mfcc = get_mfcc(save_name, flags['mfcc'])
            prediction, prediction_name = make_mfcc_prediction(model, flags, mfcc)
            res[prediction_name] += 1
            begin += flags['shift']
    return max(res.keys(), key=(lambda k: res[k])) 
