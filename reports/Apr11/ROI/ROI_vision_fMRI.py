from nilearn.input_data import NiftiMasker
import glob
import numpy as np
from nilearn.image import math_img

####################################################################

parc = '/home/sana4471/BelugaStorage/project_rrg-pbellec/sana4471/movie_decoding_sa/second_version/template_cambridge_basc_multiscale_sym_scale007.nii.gz'
visual = math_img('img==4', img=parc)
masker = NiftiMasker(standardize=True, smoothing_fwhm=8, detrend=False, mask_img=visual)
#masker = NiftiMasker(standardize=True, smoothing_fwhm=5 , detrend=True, mask_img=visual)

####################################################################

path='/home/sana4471/BelugaStorage/project_rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/**/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

configfiles = glob.glob(path, recursive=True)
configfiles=sorted(configfiles)

print(len(configfiles))

#####################################################################

fMRI_T=[]

for path in configfiles:
    print(path)
    Data_fmri = masker.fit_transform(path)
    print(Data_fmri.shape)
    fMRI_T.append(Data_fmri)

np.save('new_Voxel_Vision_F.npy', fMRI_T)
print('##################### Done!')



