"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset

/home/neu307/lml/DynaBOA/
cp -r data/human_datasets/h3.6m/training/images2/* data/human_datasets/h3.6m/training/images
/root/data/liumeilin/DynaBOA/
"""

# PW3D_ROOT = 'data/human_datasets/3dpw/'
PW3D_ROOT = '/home/neu307/liumeilin/datasets/3dpw_c/'
H36M_ROOT = '/home/neu307/liumeilin/boa_data/human_datasets/h3.6m/training'
# H36M_ROOT = '/media/milan/TOSHIBA EXT/human36m_full_raw/training'
InternetData_ROOT = 'supp_assets/bilibili'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = '/home/neu307/liumeilin/boa_data/data_extras'

JOINT_REGRESSOR_TRAIN_EXTRA = '/home/neu307/liumeilin/boa_data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = '/home/neu307/liumeilin/boa_data/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = '/home/neu307/liumeilin/boa_data/smpl_mean_params.npz'
SMPL_MODEL_DIR = '/home/neu307/liumeilin/boa_data/smpl'
