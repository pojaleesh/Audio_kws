import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM, GRU, TimeDistributed
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, MaxPool2D, MaxPool1D
from keras.backend import expand_dims, squeeze
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

def DNN_model(input_shape, nclass):
    
    input_ = Input(input_shape)
    
    X = Dense(64, activation="relu")(input_)
    X = Dense(128, activation="relu")(X)
    
    X = Flatten()(X)
    
    X = expand_dims(X, axis=-1)
    X = MaxPool1D(pool_size=2, strides=2, data_format='channels_last')(X)
    X = squeeze(X, axis=-1)
    
    X = Dropout(rate=0.1)(X)
    X = Dense(128, activation="linear")(X)
    X = Dense(256, activation="relu")(X)
    
    output_ = Dense(nclass, activation='softmax')(X)
    
    ret_model = Model(inputs = input_, outputs=output_)
    
    return ret_model
