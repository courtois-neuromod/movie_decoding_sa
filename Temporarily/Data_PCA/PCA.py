from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


fMRI_Data= np.load('Aliye1_fMRI_Data.npy', allow_pickle=True)
Movie_Data= np.load('Aliye1_Movie_Data_fp30.npy', allow_pickle=True)


pca = PCA(n_components=2000)
pca.fit(Movie_Data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Movie_Cumulative explained variance')
plt.savefig('Movie_PCA.png')

pca = PCA(n_components=444)
pca.fit(fMRI_Data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('fMRI_Cumulative explained variance')
plt.savefig('fMRI_PCA.png')



