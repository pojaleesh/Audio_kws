from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from sklearn.preprocessing import LabelEncoder

import librosa
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pickle
import numpy as np

def make_model(input_shape):
    nclass = 14
    inp = Input(input_shape)
    x = Conv2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.2)(x)
    out = Dense(nclass, activation='softmax')(x)
    ret_model = Model(inputs = inp, outputs=out)
    return ret_model

def get_result_name(idx):
    return lb.classes_[idx]

def load_data(flags):
    filename = flags['labels']
    infile = open(filename,'rb')
    lb = pickle.load(infile)
    infile.close()
    
    filename = flags['mean']
    outfile = open(filename,'rb')
    mean = pickle.load(outfile)
    outfile.close()
    
    filename = flags['std']
    outfile = open(filename,'rb')
    std = pickle.load(outfile)
    outfile.close()
    
    return lb, mean, std

predictModel = make_model((30, 216, 1))
predictModel.load_weights("Emotion_biometric_recognition/models/Main_model_1.h5")
opt = optimizers.Adam(0.001)
predictModel.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ["accuracy"])

def make_predict(flags):
    lb, mean, std = load_data(flags)
    X, sample_rate = librosa.load(flags['path'], res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30)
    if mfccs.shape[1] < 216:
        b = np.zeros((30, 216 - mfccs.shape[1]))
        mfccs = np.concatenate((mfccs, b), axis=1)
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = (mfccs - mean)/std
    mffcs = np.array(mfccs)
    mfccs = mfccs.reshape((mfccs.shape[0], mfccs.shape[1], mfccs.shape[2], 1))
    prediction = predictModel.predict(mfccs, batch_size=16, verbose=1)
    max_p = prediction.max(axis=1)
    max_index = prediction.argmax(axis=1)[0]
    print(prediction)
    return max_p, lb[max_index]
