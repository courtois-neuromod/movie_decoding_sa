from load_confounds import Params24
from nilearn.input_data import NiftiLabelsMasker
import glob
import numpy as np

####################################################################

path='/home/sana4471/projects/rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/**/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

configfiles = glob.glob(path, recursive=True)
configfiles=sorted(configfiles)

print('#################### The number of bold_files:')
print(len(configfiles))

#####################################################################

fMRI_T=[]

for path in configfiles:
    print(path)
    masker= NiftiLabelsMasker(labels_img='MIST_444.nii.gz', standardize=True, detrend=False, smoothing_fwhm=8).fit()
    Data_fmri=masker.transform(path, confounds=Params24().load(path))
    print(Data_fmri.shape)
    fMRI_T.append(Data_fmri)

np.save('Data_fMRI_Parcel.npy', fMRI_T)
print('##################### Done!')


