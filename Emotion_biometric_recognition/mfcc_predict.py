import numpy as np
import pickle


def load_data(flags):
    lb, mean, std = None, None, None
    if 'labels' in flags:
        filename = flags['labels']
        infile = open(filename,'rb')
        lb = pickle.load(infile)
        infile.close()
    if 'mean' in flags:
        filename = flags['mean']
        infile = open(filename,'rb')
        mean = pickle.load(infile)
        infile.close()
    if 'std' in flags:
        filename = flags['std']
        infile = open(filename,'rb')
        std = pickle.load(infile)
        infile.close()
    return lb, mean, std


def make_mfcc_prediction(model, flags, mfcc):
    lb, mean, std = load_data(flags['load_data'])
    mfcc = (mfcc - mean) / std
    mfcc = np.array(mfcc)
    prediction = model.predict(mfcc, batch_size=16, verbose=1)
    p = prediction.max(axis=1)
    index = prediction.argmax(axis=1)[0]
    return p, lb[index]
