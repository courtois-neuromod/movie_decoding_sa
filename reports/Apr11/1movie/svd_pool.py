import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD

#X= np.load('Data_Movie_T2.npy', allow_pickle=True)
X= np.load('Tpool_Movie.npy', allow_pickle=True)
#X= np.vstack(X)
print (X.shape)


#pca = PCA().fit(X)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance');

#pca pca = PCA(n_components=10000)
#pca= IncrementalPCA(n_components=4000, batch_size=8000)

svd = TruncatedSVD(n_components=5000, n_iter=7, random_state=42)
Y= svd.fit_transform(X)

print(Y.shape)
np.save('2svd_TT_pool.npy', Y)

