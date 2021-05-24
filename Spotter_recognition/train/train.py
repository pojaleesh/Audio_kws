import sys
import os
sys.path.append('/Users/mbassalaev/Desktop/audio_project/Spotter_recognition/')

import numpy as np
import pandas as pd
import pickle

import random
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from keras import optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.ops import gen_audio_ops as audio_ops
from sklearn.metrics import classification_report
import tensorflow as tf

from service.data_augmentation import make_batch, load_background, normalize_df
from models.cnn import CNN_model
from models.crnn import CRNN_model
from models.att_rnn import ATT_RNN_model
from models.ds_cnn import DS_CNN_model
from models.dnn import DNN_model


def prepare_data(flags):
    #../../spotter_data_service/lb
    filename = os.path.abspath(flags['label_path'])
    outfile = open(filename,'rb')
    lb = pickle.load(outfile)
    outfile.close()
    #../../../spotter_data/_background_noise_/
    background_folder = os.path.abspath(flags['background_path'])
    noise_to_data = load_background(background_folder)
    return lb, background_folder, noise_to_data 


def train_model(flags):
    df = pd.read_csv(os.path.abspath('../full_df'))
    df_train, df_test = train_test_split(
        df, shuffle=True, test_size=flags['test_size'], 
        train_size = flags['train_size'], random_state=42
    )
    
    model, opt = None, None
    train_shape = flags['train_shape']
    label_count = flags['labels']
    
    if flags['model'] == 'DNN':
        model = DNN_model(train_shape, label_count)
    elif flags['model'] == 'DS_CNN':
        model = DS_CNN_model(train_shape, label_count)
    elif flags['model'] == 'CNN':
        model = CNN_model(train_shape, label_count)
    elif flags['model'] == 'CRNN':
        model = CRNN_model(train_shape, label_count)
    elif flags['model'] == 'ATT_RNN':
        model = ATT_RNN_model(train_shape, label_count)
    else:
        raise ValueError('Unknown model for training')
    
    if flags['optimizer'] == 'Adam':
        opt = optimizers.Adam(0.001)
    elif flags['optimizer'] == 'momentum':
        opt = optimizers.SGD(momentum=0.99)
    else:
        raise ValueError('Unknown optimizer for training')
        
    model.compile(optimizer = opt, loss=CategoricalCrossentropy(from_logits=True), metrics = ["accuracy"])
    
    filename = os.path.abspath('../../spotter_data_service/lb')
    outfile = open(filename,'rb')
    lb = pickle.load(outfile)
    outfile.close()
    
    lb, background_folder, noise_to_data = prepare_data(flags)
    
    sr = flags['sr']

    p_aug = {
        'gaussian': 1,
        'stretch': 0.3,
        'pitch_shift': 0.3,
        'shift': 0.3,
        'clipping': 0.3,
        'time_mask': 0.5,
        'freq_mask': 0.5
    }

    X_test, Y_test = normalize_df(df_test, lb, sr, "tensorflow")

    start_step = 1
    training_steps_list = [20000, 20000, 20000, 20000]
    learning_rates_list = [0.001,0.0005,0.0001,0.00002]
    batch_size = flags['batch_size']
    step_interval = flags['step_interval']
    training_steps_max = np.sum(training_steps_list)
    lr_init = learning_rates_list[0]
    exp_rate = -np.log(learning_rates_list[-1] / lr_init) / training_steps_max
    
    train_metrics = []
    test_metrics = []
    
    for step in range(start_step, 100 + 1):
        print(step)
        X, Y = make_batch(df_train, lb, background_folder, noise_to_data, batch_size, sr, p_aug, "tensorflow")
        if Y.shape != (batch_size, label_count):
            raise ValueError('Invalid train batch shape')
        learning_rate_value = lr_init * np.exp(-exp_rate * step)
        K.set_value(model.optimizer.learning_rate, learning_rate_value)
        train_step_metric =  model.train_on_batch(X, Y)
        if step > 0 and step % step_interval == 0:
            if flags['debug']:
                print(f"Current step = {step}")
                print(f'Model lr = {learning_rate_value}')
            test_step_metric = model.evaluate(X_test, Y_test, batch_size=64)
            train_metrics.append(train_step_metric)
            test_metrics.append(test_step_metric)
            if flags['debug']:
                print(f"Train metrics = {train_step_metric}, Test metrics = {test_step_metric}")
    
    return model, train_metrics, test_metrics

