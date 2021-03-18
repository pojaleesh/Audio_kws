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

def DS_CNN_model(input_shape, nclass):
    
    input_ = Input(input_shape)
    
    X = Convolution2D(300, (3, 3), activation="relu", dilation_rate=(2, 1), strides=(1, 1), padding="valid")(input_)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    
    X = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(1,1), padding='valid')(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    X = Convolution2D(300, kernel_size=(1,1))(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    
    X = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(2,2), padding='valid')(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    X = Convolution2D(300, kernel_size=(1,1))(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    
    X = DepthwiseConv2D(kernel_size=(10,3), strides=(1,1), dilation_rate=(1,1), padding='valid')(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    X = Convolution2D(300, kernel_size=(1,1))(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    
    X = DepthwiseConv2D(kernel_size=(5,3), strides=(1,1), dilation_rate=(2, 2), padding='valid')(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    X = Convolution2D(300, kernel_size=(1,1))(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    
    X = DepthwiseConv2D(kernel_size=(10,3), strides=(1,1), dilation_rate=(1,1), padding='valid')(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    X = Convolution2D(300, kernel_size=(1,1))(X)
    X = BatchNormalization(momentum=0.98, center=True, scale=False, renorm=False)(X)
    X = Activation('relu')(X)
    
    X = AveragePooling2D(pool_size=(X.shape[1], X.shape[2]))(X)
    
    X = Flatten()(X)
    X = Dropout(rate=0.2)(X)
    output_ = Dense(nclass, activation='softmax')(X)
    
    ret_model = Model(inputs = input_, outputs=output_)
    
    return ret_model