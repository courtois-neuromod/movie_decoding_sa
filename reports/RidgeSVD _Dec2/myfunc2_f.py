import warnings
warnings.filterwarnings('ignore')

import sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import joblib

print("sklearn version**********")
print(sklearn.__version__)

from threadpoolctl import threadpool_info
from pprint import pprint
import os
#print('#######################################################################')
import numpy as np
print("***the current state of the threadpool-enabled runtime libraries that are loaded ")
pprint(threadpool_info())

################################################




def my_func_new_1():

    print('###############################################################################')
    print("***the current state of the threadpool-enabled runtime libraries that are loaded ")
    pprint(threadpool_info())


    #fMRI_Data= np.load('/home/sahmadi/new/last_version_Data/new_Voxel_Vision_F.npy', allow_pickle=True)
    fMRI_Data= np.load('/home/sahmadi/Subject1/Sub1_ROI_T1.npy', allow_pickle=True)
    #Movie_Data= np.load('/home/sahmadi/new/E_shift4_Movie_Data.npy', allow_pickle=True)
    
    print (Movie_Data.shape, fMRI_Data.shape)

    fMRI_Data= np.vstack(fMRI_Data)
    Movie_Data= np.vstack(Movie_Data)
    print("****************Data_shape")
    print (Movie_Data.shape, fMRI_Data.shape)


    x_train,x_test, y_train, y_test = train_test_split(Movie_Data[0:2000, :], fMRI_Data[0:2000, 0:4096], test_size=0.1)
    print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)


    start = time.time()
    print(start)
    ###########################################################





    with joblib.parallel_backend('dask'):


        R=RidgeCV(alphas=[0.1, 1, 100, 200, 300, 400, 600,  800, 900, 1000, 1200], gcv_mode='svd').fit(x_train, y_train)


    #R=RidgeCV(alphas=[0.1, 1, 100, 200, 300, 400, 600,  800, 900, 1000, 1200], gcv_mode='svd').fit(x_train, y_train)


    #########################################################
    stop = time.time()

