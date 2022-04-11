
from load_confounds import Params24
from nilearn.input_data import NiftiMasker
import glob
import numpy as np

####################################################################


mask= '/project/rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/ses-001/func/sub-01_ses-001_task-s01e01b_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'


######################################################################

configfiles = glob.glob('/home/sana4471/projects/rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/**/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', recursive=True)
configfiles=sorted(configfiles)

print(len(configfiles)) 

##################################

fMRI_T=[]

for path in configfiles: 
    print(path)
    
    masker = NiftiMasker(standardize=True, smoothing_fwhm=5 , detrend=True, mask_img=mask).fit()
    Data_fmri=masker.transform(path, confounds=Params24().load(path))   
        
    print(Data_fmri.shape)
    fMRI_T.append(Data_fmri)

np.save('voxel_wise_sub1_140.npy', fMRI_T)
print('##################### Done!')
