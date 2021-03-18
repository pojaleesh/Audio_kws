import librosa
import numpy as np

def get_mfcc(path, flags):
    X, sample_rate = librosa.load(
        path, res_type='kaiser_fast', 
        duration=flags['duration'], 
        sr=flags['sample_rate'], 
        offset=flags['offset']
    )
    sample_rate = np.array(sample_rate)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=flags['n_mfcc'])
    if mfccs.shape[1] < flags['columns']:
        b = np.zeros((flags['n_mfcc'], flags['columns'] - mfccs.shape[1]))
        mfccs = np.concatenate((mfccs, b), axis=1)
    if mfccs.shape[1] > flags['columns']:
        mfccs = mfccs[:, :flags['columns']]
    mfccs = np.expand_dims(mfccs, axis=0)
    return mfccs

def make_mfcc_data(df, flags):
    X = []
    for index,path in enumerate(df.path):  
        mfcc = get_mfcc(path, flags)
        X.append(mfcc)
        if flags['debug'] and index % 100 == 0:
            print(f'Current index = {index}')
    X = np.concatenate(X, axis=0)
    return X