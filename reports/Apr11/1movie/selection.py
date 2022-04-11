import math
import numpy as np

#Data1= np.load('Data_fMRI_voxel_f.npy', allow_pickle=True)
Data1= np.load('H_sh_conv_S1_f30.npy', allow_pickle=True)
Data2= np.load('sh_Conv_S2_f30.npy', allow_pickle=True)
Data3= np.load('H_sh_Conv_S3_f30.npy', allow_pickle=True)

Data1= np.vstack(Data1)
Data2= np.vstack(Data2)
Data3= np.vstack(Data3)


print (Data1.shape)
print (Data2.shape)
print (Data3.shape)

X_T=np.concatenate((Data1, Data2, Data3))
print('########################')
print (X_T.shape)

'''
Lable=[]
count_T=[]
count=[]        
Movie_2=[]
Movie_T2=[]


for i in range():

    count=[]
    Movie_2=[]
    #rate=len(Movie[i])/ len(fMRI[i]) 
    #102862/ 66425=1.54
    rate =1.5
    CC=np.array(Movie[i])
    for j in range (len(Movie[i])):

        if (rate*j) < len(Movie[i]):

            count.append(math.floor(rate*j))

            kk=math.floor(rate*j)
            Movie_2.append(CC[kk]) 
    count_T.append(count) 
    Movie_T2.append(Movie_2)

'''


####################################
rate=1.4857
count=[]
Movie_2=[]

Data= X_T

for j in range (len(Data)):

    if (rate*j) < len(Data):

        #count.append(math.floor(rate*j))

        kk=math.floor(rate*j)
        CC=np.array(Data[kk])
        Movie_2.append(CC)

#np.save('sh_count.npy', count)
np.save('T_CONV_Movie.npy', Movie_2)

print(Movie_2.shape)
#print(count.shape)





