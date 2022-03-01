import os
import re
import h5py
import cv2
import numpy as np
import torch
import torch.utils.data
import pytorch_lightning as pl
from tqdm import tqdm
from typing import Any, Union, List, Optional
from os.path import isdir, join
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset, DataLoader, random_split
from utils.kspace import spatial2kspace, kspace2spatial
from utils.kspace import KTransform, pt_kspace2spatial, pt_spatial2kspace


class FastMRITransform(object):

    def __init__(self, mask_f=None, target_shape=(320, 320), 
                 noise_level=0.0, noise_type='none', noise_in_range=False):
        self.mask_f = mask_f
        self.target_shape = target_shape
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.noise_in_range = noise_in_range

    @staticmethod
    def normalize(x: np.ndarray):
        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-11)
        return x, mean, std

    def sample_noise_lvl(self, noise_level, noise_in_range=False):
        if noise_in_range:
            noise_level = np.random.randint(low=0, high=noise_level + 20)
        return noise_level

    def add_noise(self, ks):
        noise_level = self.sample_noise_lvl(self.noise_level, self.noise_in_range)

        if self.noise_type == 'normal':
            ks = ks + np.random.normal(size=(ks.shape[0], ks.shape[1])) \
            * ks.mean() * noise_level
        elif self.noise_type == 'poisson':
            ks = ks + np.random.poisson(size=(ks.shape[0], ks.shape[1])) * ks.mean() * noise_level
        elif self.noise_type == 'salt':
            shape = ks.shape
            ks = ks.flatten()
            i = np.random.randint(low=0, high=shape[0] * shape[1], size=10)
            ks[i] = ks.mean() * noise_level
            ks = ks.reshape(shape)
        return ks

    def __call__(self, f_name: str, slice_id: str, k_space: np.ndarray, max_val: float, target_ks=None):

        if self.target_shape:
            xs = (k_space.shape[0] - self.target_shape[0]) // 2
            ys = (k_space.shape[1] - self.target_shape[1]) // 2
            xt = xs + self.target_shape[0]
            yt = ys + self.target_shape[1]
            k_space = k_space[xs:xt, ys:yt]
            target_ks = target_ks[xs:xt, ys:yt]

        target_k_space = torch.as_tensor(np.stack((target_ks.real, target_ks.imag)), dtype=torch.float)
        recon = kspace2spatial(k_space)
        k_space = self.add_noise(k_space)
        if self.mask_f:
            k_space, mask = apply_mask(k_space, self.mask_f)
        sampled_image = kspace2spatial(k_space)
        sampled_image, mean, std = self.normalize(sampled_image)
        if target_ks is None:
            target = (recon - mean) / (std + 1e-11)
        else:
            target = (kspace2spatial(target_ks) - mean) / (std + 1e-11)

        k_space = torch.as_tensor(np.stack((k_space.real, k_space.imag)), dtype=torch.float)
        mask = torch.as_tensor(mask, dtype=torch.float).unsqueeze(0)
        target = torch.as_tensor(target, dtype=torch.float).unsqueeze(0)
        sampled_image = torch.as_tensor(sampled_image, dtype=torch.float).unsqueeze(0)
        mean = torch.as_tensor(mean, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        std = torch.as_tensor(std, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return {'k_space': k_space,
                'target_k_space': target_k_space,
                'undersamle_mask': mask,
                'target': target,
                'sampled_image': sampled_image,
                'mean': mean,
                'std': std,
                'f_name': f_name,
                'slice_id': slice_id,
                'max_val': max_val}

    
class FastMRIh5Dataset(Dataset):

    def __init__(self, hf_path, transform: FastMRITransform):
        super(FastMRIh5Dataset, self).__init__()
        self.hf_path = hf_path
        self.hf = h5py.File(hf_path)
        self.transform = transform
        self._slices = []
        for f in sorted(list(self.hf.keys())):
            for iz in range(self.hf[f].shape[0]):
                self._slices += [(f, iz)]

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, index) -> T_co:
        key, slice_id = self._slices[index]
        ks = self.hf[key][:]
        ks = ks * 1e6
        maxval = pt_kspace2spatial(torch.from_numpy(ks) * 1e6).abs().max()
        if self.transform:
            return self.transform(key, slice_id, ks[slice_id], maxval)
        else:
            return torch.as_tensor(np.stack((ks.real, ks.imag)), dtype=torch.float)
        

class DemotionFastMRIh5Dataset(FastMRIh5Dataset):

    def __init__(self, hf_path, transform: FastMRITransform, ktransform: KTransform, z_slices=1.0):
        super(DemotionFastMRIh5Dataset, self).__init__(hf_path=hf_path, transform=transform)
        self.ktransform = ktransform
        self.z_slices = z_slices
        self._slices = []
        for f in sorted(list(self.hf.keys())):
            zc = self.hf[f].shape[0] // 2
            displ = int(zc * z_slices + 1)
            for iz in range(displ * 2):
                self._slices += [(f, iz)]

    @staticmethod
    def normalize(x: np.ndarray):
        mean = x.mean()
        std = x.std()
        x = (x - mean) / (std + 1e-11)
        return x, mean, std
    
    def create_placeholder_dict(self, k_space, f_name, slice_id, maxval,
                                target_ks):
        mask = np.ones((k_space.shape[-2:]))
        sampled_image = kspace2spatial(k_space)
        sampled_image, mean, std = self.normalize(sampled_image)
        target = (kspace2spatial(target_ks) - mean) / (std + 1e-11)

        mean = torch.as_tensor(mean, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        std = torch.as_tensor(std, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = torch.as_tensor(mask, dtype=torch.float).unsqueeze(0)
        target = torch.as_tensor(target, dtype=torch.float).unsqueeze(0)
        sampled_image = torch.as_tensor(sampled_image, dtype=torch.float).unsqueeze(0)

        k_space = torch.as_tensor(np.stack((k_space.real, k_space.imag)),
                                  dtype=torch.float)
        return {'k_space': k_space,
                'undersamle_mask': mask,
                'target': target,
                'sampled_image': sampled_image,
                'mean': mean,
                'std': std,
                'f_nase': f_name,
                'slice_id': slice_id,
                'max_val': maxval} 
        
    def __getitem__(self, item):
        key, slice_id = self._slices[item]
        k_space = torch.from_numpy(self.hf[key][slice_id]).unsqueeze(0)
        new_k_space, rot_vector, shift_vector = self.ktransform(k_space.clone())
        maxval = pt_kspace2spatial(torch.from_numpy(self.hf[key][slice_id])).abs().max()
        
        if self.transform:
            batch = self.transform(key, slice_id, new_k_space[0].numpy(),
                                   maxval, target_ks=k_space[0].numpy())
        else:
            batch = self.create_placeholder_dict(new_k_space[0].numpy(),
                                                 key, slice_id, maxval,
                                                 k_space[0])
            batch['target_k_space'] = torch.as_tensor(np.stack((k_space[0].real,
                                                                k_space[0].imag)),
                                                      dtype=torch.float)
        batch['rot_vector'] = rot_vector
        batch['shift_vector'] = shift_vector
        return batch
