print("Sana")

import warnings
warnings.filterwarnings('ignore')


import numpy as np

np.show_config()

import time
import npp
#from sklearn.decomposition import PCA
#from ridge import bootstrap_ridge
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV


fMRI_Data= np.load('fMRI_Data_subject3.npy', allow_pickle=True)
Movie_Data= np.load('Movie_Data_subject3.npy', allow_pickle=True)

print (Movie_Data.shape, fMRI_Data.shape)



x_train,x_test, y_train, y_test = train_test_split(Movie_Data, fMRI_Data , test_size=0.1)
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)


start = time.time()
###########################################################

clf = RidgeCV(alphas=[1.000, 2.783, 7.743, 21.544, 59.948, 166.810, 464.159, 1291.550, 3593.814, 10000.000]).fit(x_train, y_train)


#########################################################
stop = time.time()
print(f"**********************************Training time: {stop - start}s")


pred_test = clf.predict(x_test)

Ridge_correlations = npp.mcorr(y_test, pred_test)

print(Ridge_correlations.shape)



#np.save('1Ridge_correlations.npy', Ridge_correlations)

#np.save('1y_test.npy', y_test)




