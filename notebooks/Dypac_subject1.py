import numpy as np
from nilearn.interfaces.fmriprep import load_confounds_strategy
from dypac.masker import LabelsMasker, MapsMasker

####################################################################

path='/home/sana4471/BelugaStorage/project_rrg-pbellec/sana4471/movie_decoding_sa/data/friends/derivatives/fmriprep-20.2lts/fmriprep/sub-01/**/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

configfiles = glob.glob(path, recursive=True)
configfiles=sorted(configfiles)

print('#################### The number of bold_files:')
print(len(configfiles))

#####################################################################

file_dypac = 'sub-01_space-MNI152NLin2009cAsym_desc-dypac256_components.nii.gz'
file_mask= 'sub-01_space-MNI152NLin2009cAsym_label-GM_mask.nii.gz'

fMRI_T=[]

for path in configfiles:
    print(path)

    conf = load_confounds_strategy(path, denoise_strategy='simple', global_signal='basic')
    masker = NiftiMasker(standardize=True, detrend=False, smoothing_fwhm=5, mask_img=file_mask)
    masker.fit(path)
    maps_masker = MapsMasker(masker=masker, maps_img=file_dypac)

    Data_fmri= maps_masker.transform(img=path, confound=conf[0])
    print(Data_fmri.shape)
    fMRI_T.append(Data_fmri)


np.save('Data_fMRI_Dypac.npy', fMRI_T)
print('##################### Done!')

