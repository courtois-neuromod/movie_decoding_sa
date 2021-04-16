

def fMRI_Data_pre (configfiles, labels_img):
    
    fMRI_T=[] 

    for path in configfiles:
        print(path)
        masker= NiftiLabelsMasker(labels_img='MIST_444.nii.gz', standardize=True, detrend=False, smoothing_fwhm=8).fit()
        Data_fmri=masker.transform(path, confounds=Params24().load(path))
        print(Data_fmri.shape)
        fMRI_T.append(Data_fmri)
        
        
    np.save('fMRI_Data_v1.npy', fMRI_T)
    print('##################### fMRI Data preparation is Done!')    
        
    return fMRI_T


def Movie_Data_pre (configfiles, model):
    
    list_f_T=[]

    for name in movie_ses_1: 
        print(name) 


        
        ################################################ frame selection_1
        
        

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
        print (" Frame extraction is Done!")


       #####################################


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

        
        ################################################ feature extraction
          


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


    np.save('Movie_Data_v1.npy', list_f_T)
    print('##################### Movie Data preparation is Done!') 
    
    return list_f_T
    
def Frame_selection(Movie_Data, fMRI_Data):


    Lable=[]
    count_T=[]
    count=[]        
    Movie_2=[]
    Movie_T2=[]

    for i in range(48):
        count=[]
        Movie_2=[]
        rate=len(Movie[i])/ len(fMRI[i]) 
        CC=np.array(Movie[i])
        for j in range (len(Movie[i])):

            if (rate*j) < len(Movie[i]):

                count.append(math.floor(rate*j))

                kk=math.floor(rate*j)
                Movie_2.append(CC[kk]) 
        count_T.append(count) 
        Movie_T2.append(Movie_2)
        
    return   Movie_T2, count_T


def make_delayed(stim, delays, circpad=False):
    
    """
    Ref:Dr Alex Huth_tutorial
    
    Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
        
    Delay_Movie_Data=np.hstack(dstims)    
    np.save('Delay_Movie_Data.npy', Delay_Movie_Data)
    return Delay_Movie_Data