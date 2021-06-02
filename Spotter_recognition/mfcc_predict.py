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
    prediction = model.predict(mfcc, batch_size=64, verbose=1)
    prediction = prediction[0]
    index_1, index_2 = prediction.argsort(axis=0)[-2:]
    prediction.sort(axis=0)
    prediction_1, prediction_2 = prediction[-2:]
    return prediction_2, lb[index_2], prediction_1, lb[index_1]

