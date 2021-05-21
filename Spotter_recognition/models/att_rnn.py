import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import (
    Dense, Embedding, LSTM, Input, Flatten, 
    Dropout, Activation, BatchNormalization, Convolution2D, 
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, MaxPool2D,
    Dot, Softmax, Bidirectional, GRU
)
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

def ATT_RNN_model(input_shape, nclass):
    
    input_ = Input(input_shape)
    
    X = Convolution2D(10, (5, 1), activation="relu", dilation_rate=(1, 1), strides=(1, 1))(input_)
    X = BatchNormalization()(X)
    X = Convolution2D(1, (5, 1), activation="relu", dilation_rate=(1, 1), strides=(1, 1))(X)
    X = BatchNormalization()(X)
    shape = X.shape
    X = tf.keras.layers.Reshape((-1, shape[2] * shape[3]))(X)
    X = Bidirectional(GRU(units=128,  return_sequences=True, unroll=True))(X)
    X = Bidirectional(GRU(units=128,  return_sequences=True, unroll=True))(X)
    
    feature_dim = X.shape[-1]
    middle = X.shape[1] // 2
    mid_feature = X[:, middle, :]
    
    query = Dense(feature_dim)(mid_feature)
        
    att_weights = Dot(axes=[1, 2])([query, X])
    att_weights = Softmax(name='attSoftmax')(att_weights)
    X = Dot(axes=[1, 1])([att_weights, X])
    X = Dropout(rate=0.1)(X)
    
    X = Dense(64, activation="relu")(X)
    X = Dense(32, activation="linear")(X)
    
    output_ = Dense(nclass, activation='softmax')(X)
    
    ret_model = Model(inputs = input_, outputs=output_)
    
    return ret_model