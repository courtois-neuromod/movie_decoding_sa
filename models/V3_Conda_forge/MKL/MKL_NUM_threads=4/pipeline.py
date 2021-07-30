print("Sana")

import os
print("******************description of thread-pools initialized*****************")
os.system('python -m threadpoolctl -i numpy scipy.linalg')


import warnings
warnings.filterwarnings('ignore')

import time
import npp
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

from threadpoolctl import threadpool_info
from pprint import pprint





###########################################
import numpy as np
print("****************the current state of the threadpool-enabled runtime libraries that are loaded ")
pprint(threadpool_info())
#print("****************Config information of numpy")
#np.show_config()

################################################


fMRI_Data= np.load('fMRI_Data_subject3.npy', allow_pickle=True)
Movie_Data= np.load('Movie_Data_subject3.npy', allow_pickle=True)

print("****************Data_shape")
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




