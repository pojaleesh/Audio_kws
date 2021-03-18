import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, MaxPool2D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

def CNN_model(input_shape, nclass):
    
    input_ = Input(input_shape)
    
    X = Convolution2D(16, (3, 3), activation="relu", dilation_rate=(1, 1), strides=(1, 1), padding="same")(input_)
    X = Convolution2D(16, (5, 3), activation="relu", dilation_rate=(1, 1), strides=(1, 1), padding="same")(X)
    
    X = GRU(units = 256)(X)
    
    X = Flatten()(X)
    X = Dropout(rate=0.1)(X)
    
    X = Dense(128, activation="linear")(X)
    X = Dense(256, activation="relu")(X)
    
    output_ = Dense(nclass, activation='softmax')(X)
    
    ret_model = Model(inputs = input_, outputs=output_)
    
    return ret_model