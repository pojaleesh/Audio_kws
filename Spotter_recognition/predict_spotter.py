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
import import_ipynb
from mfcc_predict import make_mfcc_prediction


def insert_one(indicator, index):
    for i in range(index, min(indicator.shape[1], index + 51)):
        indicator[0, i] = 1
    return indicator


def make_predict(model, flags, threshold):
    
    path = flags['path_name']
    
    data = AudioSegment.from_wav(path)
    indicator = np.zeros((1, len(data)))
    print(len(data))
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
            prediction, prediction_name = make_mfcc_prediction(flags, mfcc)
            print(prediction, prediction_name)
            temp.append(prediction)
            if prediction_name != 'unknown' and prediction > threshold:
                indicator = insert_one(indicator, begin)
            begin += flags['shift']
            
    return indicator, temp