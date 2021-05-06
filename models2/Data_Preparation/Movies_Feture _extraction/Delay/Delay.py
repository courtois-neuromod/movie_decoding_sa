import numpy as np
from sklearn.model_selection import train_test_split


fMRI_Data= np.load('Aliye1_fMRI_Data.npy', allow_pickle=True)
Movie_Data= np.load('Aliye1_Movie_Data_fp30.npy', allow_pickle=True)

x_train, x_test, y_train, y_test = train_test_split( Movie_Data, fMRI_Data, test_size=0.1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print('####################################')

def make_delayed(stim, delays, circpad=False):
    """
    Ref:Dr Alex Huth_tutorial 
    
    Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)


delays = [1,2,3,4] # Delay vector [1,2,3,4] increses the feature vestor size 4096 to 16,384 
# Delay vector [1,2,3] increses the feature vestor size 4096 to 12,228 
del_training_features = make_delayed(x_train, delays)
del_test_features = make_delayed(x_test, delays)


print(del_training_features.shape,del_test_features.shape)

np.save('del_training_features2.npy', del_training_features)
np.save('del_test_features2.npy', del_test_features)
np.save('y_train2.npy', y_train)
np.save('y_test2.npy', y_test)



