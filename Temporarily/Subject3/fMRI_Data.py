from load_confounds import Params24
from nilearn.input_data import NiftiLabelsMasker
import glob
import numpy as np

####################################################################

configfiles = glob.glob('/home/sana4471/projects/rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.1.0/fmriprep/sub-03/**/*preproc_bold.nii.gz', recursive=True)
configfiles=sorted(configfiles)

print('#################### The number of BOLD files:')
print(len(configfiles)) 

#####################################################################

fMRI_T=[]

for path in configfiles: 
    print(path)
    masker= NiftiLabelsMasker(labels_img='MIST_444.nii.gz', standardize=True, detrend=False, smoothing_fwhm=8).fit()
    Data_fmri=masker.transform(path, confounds=Params24().load(path))
    print(Data_fmri.shape)
    fMRI_T.append(Data_fmri)

np.save('fMRI_Data_sub3.npy', fMRI_T)
print('##################### Done!')

