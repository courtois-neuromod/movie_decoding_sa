import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
import numpy as np
import glob
import shutil
from tensorflow.keras.models import Model



Video_path='/home/sana4471/projects/rrg-pbellec/sana4471/movie_decoding_sa/data/friends/stimuli/s1/*.mkv'

movie_ses_1=sorted(glob.glob(Video_path) ) 



base_model=VGG16()
base_model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


list_f_T=[]

for name in movie_ses_1: 
    print(name) 
     
        
        
    vidcap = cv2.VideoCapture(name)
    count = 0
    success = True
    while success:
        success,frame = vidcap.read()
        #print('read frame %d:' %count,success)
        fps=30
        Rate=1 # we can cover all scense of movie. 
        #if math.floor(count%(Rate*23)) == 0 : 
        if count%(Rate*fps) == 0 :   # using this condition we can determine the number of frames 

            cv2.imwrite('frames/frame %d.jpg'%count,frame)
            #print('successfully written the frame_Rate1 %d'%count)
        count+=1
    print ("Done!")
    
    
    
    
    
    list_n=[]
    list_f=[]
    
    path='frames'

    filenames2 = glob.glob(path + "/*.jpg") #read all files in the path mentioned
    
    


    for x in filenames2:

        s=x

        start = s.find('/frame')+7 
        end = s.find('.')
        x_p=s[start:end]
        #print(x_p)

        list_n.append(x_p)
        #i=i+1

    list_n.sort(key=int)
    
    

    
    for i in range (len(list_n)-1):
        
        
        img_path = 'frames/'+ 'frame '+ list_n[i]+ '.jpg'
        #print(img_path)
        
        img = image.load_img(img_path, target_size=(224, 224))
        x_img = image.img_to_array(img)
        x_img = np.expand_dims(x_img, axis=0)
        x_img = preprocess_input(x_img)


        avg_pool_features = model.predict(x_img)
        features_reduce =  avg_pool_features.squeeze()
        
        #print("Feture extraction is done")
   
        list_f.append(features_reduce) 
        
        
    list_f_T.append(list_f)
    
    
    frame_features_s = np.array(list_f)
    print(frame_features_s.shape)
    
    
    
    files = glob.glob('frames/*')
    for f in files:
        os.remove(f) 
                       

np.save('Shahla_correct_Sea11_f30.npy', list_f_T)
#print('##########   find the problem_14b_Sea1  ################DONE!')

