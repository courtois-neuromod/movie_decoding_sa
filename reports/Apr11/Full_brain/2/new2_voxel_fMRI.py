from load_confounds import Params36
from nilearn.input_data import NiftiMasker
import glob
import numpy as np

####################################################################

#sub-01_ses-001_task-s01e01b_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
#sub-01_ses-001_task-s01e01a_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
#path=/home/sana4471/project/rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/ses-001/func

#mask= '/project/rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/ses-001/func/sub-01_ses-001_task-s01e01b_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
mask= '/home/sana4471/BelugaStorage/project_rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/ses-001/func/sub-01_ses-001_task-s01e01b_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

#/home/sana4471/BelugaStorage/project_rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/ses-001/func

#######################################################################################################################################

#configfiles = glob.glob('/home/sana4471/projects/rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/**/*preproc_bold.nii.gz', recursive=True)
configfiles = glob.glob('/home/sana4471/BelugaStorage/project_rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/**/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', recursive=True)

configfiles=sorted(configfiles)

print('#################### The number of preproc_bold files :')
print(len(configfiles)) 

##################################

fMRI_T=[]

for path in configfiles: 
    print(path)
    
    masker = NiftiMasker(mask_img=mask, standardize=True, detrend=False, smoothing_fwhm=8).fit()
    Data_fmri=masker.transform(path, confounds=Params36().load(path))   
        
    print(Data_fmri.shape)
    fMRI_T.append(Data_fmri)

np.save('new_par36_voxel_wise_sub1_F.npy', fMRI_T)
print('##################### Done!')
