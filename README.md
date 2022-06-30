# AUTOFOCUSING-PLUS
Official MICCAI-2022 accepted paper repository

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

## Citation
```
@misc{AFPlusMRI,
  doi = {10.48550/ARXIV.2203.05569},
  url = {https://arxiv.org/abs/2203.05569},
  author = {Kuzmina, Ekaterina and Razumov, Artem and Rogov, Oleg Y. and Adalsteinsson, Elfar and White, Jacob and Dylov, Dmitry V.},
  keywords = {Image and Video Processing (eess.IV), Artificial Intelligence (cs.AI), Computer Vision and Pattern Recognition (cs.CV), Medical Physics (physics.med-ph), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Physical sciences, FOS: Physical sciences},
  title = {Autofocusing+: Noise-Resilient Motion Correction in Magnetic Resonance Imaging},
  publisher = {arXiv},
  year = {2022},
}

```
