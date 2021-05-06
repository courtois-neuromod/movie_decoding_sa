import numpy as np

import matplotlib.pyplot as plt

y_test= np.load('y_test2.npy', allow_pickle=True)
y_train= np.load('y_train2.npy', allow_pickle=True)
x_train= np.load('del_training_features2.npy', allow_pickle=True)
x_test= np.load('del_test_features2.npy', allow_pickle=True)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

########################################################################

beta_ols = np.linalg.lstsq(x_train, y_train)[0]
print(beta_ols.shape)

pred_test = x_test.dot(beta_ols)

import npp

ols_correlations = npp.mcorr(y_test, pred_test)

print(ols_correlations.shape)


plt.hist(ols_correlations, 50)
plt.xlim(-1, 1)
plt.xlabel("Linear Correlation")
plt.ylabel("Num.Parcells");

plt.savefig('Linear Correlation.png')

np.save('ols_correlations.npy', ols_correlations )




