import numpy as np

X= np.load('2svd_TT_pool.npy', allow_pickle=True)
print (X.shape)


def make_delayed(stim, delays, circpad=False):

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

    Delay_Movie_Data=np.hstack(dstims)
    np.save('E_pool_shift4_Movie_Data.npy', Delay_Movie_Data)
    print(Delay_Movie_Data.shape)
    return Delay_Movie_Data

delays = [1,2,3,4]
del_training_features = make_delayed(X, delays)

