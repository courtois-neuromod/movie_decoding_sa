from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.input_data import NiftiMasker
import glob
import numpy as np

####################################################################

mask= '/home/sana4471/BelugaStorage/project_rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/ses-001/func/sub-01_ses-001_task-s01e01b_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

######################################################################

configfiles = glob.glob('/home/sana4471/BelugaStorage/project_rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/**/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', recursive=True)
configfiles=sorted(configfiles)

print(len(configfiles)) 


masker = NiftiMasker(standardize=True, smoothing_fwhm=5 , detrend=True, mask_img=mask).fit()

##################################

fMRI_T=[]

for path in configfiles: 
    print(path)
    

    conf = load_confounds_strategy(path, denoise_strategy='simple', global_signal='basic')
    Data_fmri=masker.transform(path, confounds=conf[0])   
        
    print(Data_fmri.shape)
    fMRI_T.append(Data_fmri)

np.save('con_str1_wise_sub1.npy', fMRI_T)
print('##################### Done!')
