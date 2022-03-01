from collections import namedtuple

SysPath = namedtuple('SysPath', ['SAVING_PATH',
                                 'MODEL_AF_PATH',
                                 'MODEL_UNET_PATH',
                                 'DATASET_PATH',
                                 'GRADMC_PATH',
                                 'NUFFT_PATH',
                                 'TRAIN_PATH',
                                 'VAL_PATH',
                                 'LOG_PATH',
                                 'TRAIN_NAME',
                                 'VAL_NAME'])

PATH = SysPath(SAVING_PATH = 'test/',
               MODEL_AF_PATH = 'model_weights/af_plus_state_dict.pt',
               MODEL_UNET_PATH = 'model_weights/unet_state_dict.pt', 
               DATASET_PATH = '/home/a_razumov/smbmount_a_razumov/fastMRIdatasets/singlecoil_val/',
               
               GRADMC_PATH = '/home/a_razumov/opt/GradMC2/',
               NUFFT_PATH = '/home/ekuzmina/AUTOFOCUSING-PLUS/pytorch_nufft',
               LOG_PATH = 'runs/',
               
               TRAIN_PATH = '/home/a_razumov/small_datasets/small_fastMRIh5_PD_3T/',
               VAL_PATH = '/home/ekuzmina/fastmri-demotion/datasets/',
               TRAIN_NAME = 'train_small_PD_3T.h5',
               VAL_NAME = 'val_1T_PD.h5'
              )