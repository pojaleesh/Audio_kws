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
    if flags['mfcc_type'] == 'librosa':
        mfcc = (mfcc - mean) / std
        mfcc = np.array(mfcc)
    prediction = model.predict(mfcc, verbose=1)
    max_p = prediction.max(axis=1)
    max_index = prediction.argmax(axis=1)[0]
    return max_p, lb[max_index]

