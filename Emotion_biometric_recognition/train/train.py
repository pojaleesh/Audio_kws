import sys
import os
sys.path.append('/Users/mbassalaev/Desktop/audio_project/Emotion_biometric_recognition/')

import numpy as np
import pandas as pd
import pickle

import random
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from keras import optimizers
from keras.utils import np_utils, to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf

from models.cnn import CNN_model_1, CNN_model_2

def train_model(flags):
    df = pd.read_csv(flags['df_path'])
    df = df.drop(columns=['Unnamed: 0'], axis=1)

    model, opt = None, None
    train_shape = flags['train_shape']
    label_count = flags['labels']

    if flags['model'] == 'CNN_1':
        model = CNN_model_1(train_shape, label_count)
    elif flags['model'] == 'CNN_2':
        model = CNN_model_2(train_shape, label_count)
    else:
        raise ValueError('Unknown model for training')

    if flags['optimizer'] == 'Adam':
        opt = optimizers.Adam(0.001)
    elif flags['optimizer'] == 'Momentum':
        opt = optimizers.SGD(momentum=0.99)
    else:
        raise ValueError('Unknown optimizer for training')

    model.compile(optimizer = opt, loss=CategoricalCrossentropy(from_logits=True), metrics = ["accuracy"])
    
    filename = flags['train_data_path'] 
    outfile = open(filename,'rb')
    X = pickle.load(outfile)
    outfile.close()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        df.labels,
        test_size=0.08,
        shuffle=True,
        random_state=42
    )

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    lb = LabelEncoder()
    Y_train = np_utils.to_categorical(lb.fit_transform(Y_train))
    Y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    start_step = 1
    training_epochs_list = [10, 20, 10, 10]
    learning_rates_list = [0.001,0.0001,0.00001,0.000001]
    batch_size = flags['batch_size']

    train_metrics = {
        'accuracy': [],
        'loss': []
    }
    test_metrics = {
        'accuracy': [],
        'loss': []
    }

    for lr, epochs in zip(learning_rates_list, training_epochs_list):
        K.set_value(model.optimizer.learning_rate, lr)
        model_history = model.fit(
            x=X_train, 
            y=Y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_test, Y_test),
            verbose=flags['debug']
        )
        train_metrics['accuracy'] += model_history.history['accuracy']
        test_metrics['accuracy'] += model_history.history['val_accuracy']
        train_metrics['loss'] += model_history.history['loss']
        test_metrics['loss'] += model_history.history['val_loss']
            
    return model, train_metrics, test_metrics

