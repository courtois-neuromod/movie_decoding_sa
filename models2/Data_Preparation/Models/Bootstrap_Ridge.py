import numpy as np
import matplotlib.pyplot as plt

y_test= np.load('y_test2.npy', allow_pickle=True)
y_train= np.load('y_train2.npy', allow_pickle=True)
x_train= np.load('del_training_features2.npy', allow_pickle=True)
x_test= np.load('del_test_features2.npy', allow_pickle=True)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


########################################################################

from ridge import bootstrap_ridge
import logging

logging.basicConfig(level=logging.INFO)

alphas = np.logspace(0, 2, 10) # Equally log-spaced alphas between 10 and 1000

wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(x_train, y_train, 
                                                     x_test, y_test,
                                                     alphas, nboots=1, chunklen=40, nchunks=20,
                                                    singcutoff=1e-10, single_alpha=True)


#################################################################################

print(wt.shape)

pred_test = x_test.dot(wt)

import npp

Ridge_correlations = npp.mcorr(y_test, pred_test)

print(Ridge_correlations.shape)

plt.hist(Ridge_correlations, 50)
plt.xlim(-1, 1)
plt.xlabel("Ridge Correlation")
plt.ylabel("Num. Parcels");

plt.savefig('Ridge Correlation.png')
np.save('Ridge_correlations.npy', Ridge_correlations)
