from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


fMRI_Data= np.load('Aliye1_fMRI_Data.npy', allow_pickle=True)
Movie_Data= np.load('Aliye1_Movie_Data_fp30.npy', allow_pickle=True)

pca = PCA(n_components=444)
fMRI_PCA = pca.fit_transform(fMRI_Data)


pca = PCA(n_components=1200)
Movie_PCA = pca.fit_transform(Movie_Data)

np.save('fMRI_PCA.npy', fMRI_PCA)
np.save('Movie_PCA_1200.npy', Movie_PCA)



