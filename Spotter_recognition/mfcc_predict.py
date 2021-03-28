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
import tensorflow as tf
import IPython.display as ipd
import random
from pydub import AudioSegment
from Spotter_recognition.service.Mfcc import get_mfcc

def load_data(flags):
    filename = flags['labels']
    infile = open(filename,'rb')
    lb = pickle.load(infile)
    infile.close()
    filename = flags['mean']
    infile = open(filename,'rb')
    mean = pickle.load(infile)
    infile.close()
    filename = flags['std']
    infile = open(filename,'rb')
    std = pickle.load(infile)
    infile.close()
    return lb, mean, std

def make_mfcc_prediction(model, flags, mfcc):
    lb, mean, std = load_data(flags)
    mfcc = (mfcc - mean) / std
    mfcc = np.array(mfcc)
    prediction = model.predict(mfcc, batch_size=16, verbose=1)
    max_p = prediction.max(axis=1)
    max_index = prediction.argmax(axis=1)[0]
    return max_p, lb[max_index]
