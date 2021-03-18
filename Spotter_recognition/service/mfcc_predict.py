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
import import_ipynb
from models.cnn import CNN_model
from service.Mfcc import get_mfcc

filename = 'labels_spotter'
infile = open(filename,'rb')
lb = pickle.load(infile)
infile.close()

model_cnn = tf.keras.models.load_model('saved_models/CNN')
model_crnn = tf.keras.models.load_model('saved_models/CRNN')
model_at_rnn = tf.keras.models.load_model('saved_models/AT_RNN')
model_ds_cnn = tf.keras.models.load_model('saved_models/DS_CNN')

def mfcc_predict(flags, mfcc):
    mfcc = (mfcc - flags['mean']) / flags['std']
    mfcc = np.array(mfcc)
    if flags['model'] == 'CNN':
        prediction = model_cnn.predict(mfcc, batch_size=16, verbose=1)
    elif flags['model'] == 'CRNN':
        prediction = model_crnn.predict(mfcc, batch_size=16, verbose=1)
    elif flags['model'] == 'AT_RNN':
        prediction = model_at_rnn.predict(mfcc, batch_size=16, verbose=1)
    elif flags['model'] == 'DNN':
        prediction = model_dnn.predict(mfcc, batch_size=16, verbose=1)
    elif flags['model'] == 'DS_CNN':
        prediction = model_dc_cnn.predict(mfcc, batch_size=16, verbose=1)
    max_p = prediction.max(axis=1)
    max_index = prediction.argmax(axis=1)
    return max_p, lb.classes_[max_index][0]