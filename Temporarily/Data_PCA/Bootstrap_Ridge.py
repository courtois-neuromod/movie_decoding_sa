import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

y= np.load('fMRI_PCA.npy', allow_pickle=True)
X= np.load('Movie_PCA_1200.npy', allow_pickle=True)

x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


########################################################################

from ridge import bootstrap_ridge
import logging

logging.basicConfig(level=logging.INFO)

alphas = np.logspace(0, 3, 10) # Equally log-spaced alphas between 10 and 1000

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
plt.xlabel("PCA_Ridge Correlation_1200")
plt.ylabel("Num. Parcels");

plt.savefig('PCA_Ridge_Correlation_1200.png')
np.save('PCA_Ridge_correlations_1200.npy', Ridge_correlations)
