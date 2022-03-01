# AUTOFOCUSING-PLUS
Official MICCAI-2022 submission repository

Matlab implementation of GradMC: http://mloss.org/software/view/430/
NUFFT implementation: https://github.com/tomer196/PILOT


## Before Running and Prerequisites

1. Firstly, you have to download FastMRI dataset from https://fastmri.org/dataset/

2. Then, add all system pathes to config.py

3. Create a datasets from FastMRI and Corrupted dataset in Dataset_creation.ipynb

4. Pre-trained weight of U-Net for Autofocusing+ and U-Net: 
https://drive.google.com/drive/folders/1NmM-ilfa0c52-c0m_Ul2OOJYZLJN5HtV?usp=sharing


For metric calculation *piq* library is used: https://github.com/photosynthesis-team/piq



## Autofocusing+ Train and Validation

- File for training of **Autofocusing+** algorithm *autofocusing_plus_train.py*

- Validation of **Autofocusing+** and other algorithms is in *validate_all_methods.py*
