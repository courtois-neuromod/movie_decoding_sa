import numpy as np
import npp
import matplotlib.pyplot as plt
from nilearn.plotting  import view_img 
from nilearn.input_data import NiftiLabelsMasker

fMRI_Data1= np.load('fMRI_label1.npy', allow_pickle=True)
fMRI_Data2= np.load('fMRI_label2.npy', allow_pickle=True)
fMRI_Data3= np.load('fMRI_label3.npy', allow_pickle=True)
fMRI_Data4= np.load('fMRI_label4.npy', allow_pickle=True)
fMRI_Data5= np.load('fMRI_label5.npy', allow_pickle=True)
fMRI_Data6= np.load('fMRI_label6.npy', allow_pickle=True)

Movie_Data1= np.load('Xx1.npy', allow_pickle=True)
Movie_Data2= np.load('Xx2.npy', allow_pickle=True)
Movie_Data3= np.load('Xx3.npy', allow_pickle=True)
Movie_Data4= np.load('Xx4.npy', allow_pickle=True)
Movie_Data5= np.load('Xx5.npy', allow_pickle=True)
Movie_Data6= np.load('Xx6.npy', allow_pickle=True)

Xx_T=np.vstack((Movie_Data1,Movie_Data2,Movie_Data3,Movie_Data4,Movie_Data5,Movie_Data6))

print('Xx_T.shape', Xx_T.shape)

#################################################

Label_T=np.vstack((fMRI_Data1,fMRI_Data2,fMRI_Data3,fMRI_Data4,fMRI_Data5,fMRI_Data6))
print('Label_T.shape', Label_T.shape)

###############################################################################################


Movie_PCA=np.load('x_test.npy', allow_pickle=True)
y_test=np.load('y_test.npy', allow_pickle=True)

Movie_Data= Movie_PCA

T1=[]
T2=[]
T3=[]
T4=[]
T5=[]
T6=[]
T7=[]
T8=[]



for i in range(len(Movie_Data)):

    if i % 6==0:
        T1.append(Movie_Data[i,:])

    if i % 6==1:
        T2.append(Movie_Data[i,:])

    if i % 6==2:
        T3.append(Movie_Data[i,:])

    if i % 6==3:
        T4.append(Movie_Data[i,:] )


    if i % 6==4:
        T5.append(Movie_Data[i,:])

    if i % 6==5:
        T6.append(Movie_Data[i,:])

T1_D= np.array(T1[0:733])
T2_D= np.array(T2[0:733])
T3_D= np.array(T3)
T4_D= np.array(T4)
T5_D= np.array(T5)
T6_D= np.array(T6)

print(T1_D.shape, T2_D.shape, T3_D.shape,T4_D.shape,T5_D.shape, T6_D.shape)

Xx_test=np.hstack((T1_D,T2_D,T3_D,T4_D,T5_D,T6_D))

##########################################################################

fMRI_Data= y_test

fMRI_label=[]

###

for i in range(len(fMRI_Data)):


    if i % 6==5:
        fMRI_label.append(fMRI_Data[i,:])


label_test= np.array(fMRI_label)

#############################################################################

print('Xx_test.shape and label_test.shape', Xx_test.shape, label_test.shape)

from ridge import bootstrap_ridge
import logging

logging.basicConfig(level=logging.INFO)


alphas = np.logspace(0, 3, 10) # Equally log-spaced alphas between 10 and 1000

wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(Xx_T[0:5000], Label_T[0:5000],
                                                     Xx_test, label_test,
                                                     alphas, nboots=1, chunklen=40, nchunks=20,
                                                    singcutoff=1e-10, single_alpha=True)


#np.save('wt.npy', wt)

#print('wt.shape', wt.shape)


###########################################################################



pred_test = Xx_test.dot(wt)

#np.save('pred_test.npy', pred_test)
#np.save('label_test.npy',label_test )
#print(pred_test.shape, label_test.shape)

Ridge_correlations = npp.mcorr(label_test, pred_test)

print('Ridge_correlations.shape', Ridge_correlations.shape)

np.save('Ridge_correlations.npy', Ridge_correlations)


plt.hist(Ridge_correlations, 50)
plt.xlim(-1, 1)
plt.xlabel("PCA_Ridge Correlation_1200")
plt.ylabel("Num. Parcels")
plt.savefig('Sub3.png')
