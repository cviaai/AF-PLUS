import math
import random
import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import contextlib
from typing import Optional, Sequence, Tuple, Union, List

from config import PATH
from utils.optimization_losses import tsavgol

import sys
sys.path.append(PATH.NUFFT_PATH)
import nufft

from torch.fft import fftshift, ifftshift, fftn, ifftn
Ft = lambda x : fftshift(fftn(ifftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
IFt = lambda x : ifftshift(ifftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)

            
def normalize(x):
    x1 = x - x.min()
    return x1 / x1.max()


def sample_noise(motion_vector, noise_lvl):
    n = torch.randn(size=motion_vector.shape) * motion_vector.mean()
    return n * noise_lvl


def add_noise(ks, noise_level, noise_in_range=False):
    if noise_in_range:
        noise_level = torch.randint(low=0, high=noise_level + 20, size=(1, ))
    ks = ks + torch.randn(size=ks.shape) * ks.mean() * noise_level
    return ks

            
def spatial2kspace(img: np.ndarray) -> np.ndarray:
    img = np.fft.ifftshift(img)
    k_space = np.fft.fftn(img, norm='ortho')
    return np.fft.fftshift(k_space)


def pt_spatial2kspace(img: torch.Tensor) -> torch.Tensor:
    img = torch.fft.ifftshift(img, dim=(-1, -2))
    k_space = torch.fft.fftn(img, dim=(-1, -2), norm='ortho')
    return torch.fft.fftshift(k_space, dim=(-1, -2))


def kspace2spatial(k_space: np.ndarray) -> np.ndarray:
    recon = np.fft.fftshift(k_space)
    recon = np.fft.ifftn(recon, norm='ortho')
    return np.abs(np.fft.ifftshift(recon))


def pt_kspace2spatial(k_space: torch.Tensor) -> torch.Tensor:
    recon = torch.fft.fftshift(k_space, dim=(-1, -2))
    recon = torch.fft.ifftn(recon, dim=(-1, -2), norm='ortho')
    return torch.fft.ifftshift(recon, dim=(-1, -2))


def get_rot_mat(theta):
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)]])


def get_rot_mat_nufft(rot_vector):
    rot_mat = torch.zeros(rot_vector.shape[0], 2, 2)#.cuda()
    rot_mat[:, 0, 0] = torch.cos(rot_vector)
    rot_mat[:, 0, 1] = -torch.sin(rot_vector)
    rot_mat[:, 1, 0] = torch.sin(rot_vector)
    rot_mat[:, 1, 1] = torch.cos(rot_vector)
    return rot_mat


def _rot_img(x, theta):
    rot_mat = _get_rot_mat(theta)[None, ...].repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def _shear_rot_img(img, theta):
    theta = torch.tensor(theta)
    s = torch.sin(theta)
    t = -torch.tan(theta / 2)
    # Sample grid
    grid = torch.stack([torch.arange(-img.shape[1] // 2, img.shape[1] // 2).float(),
                        torch.arange(-img.shape[2] // 2, img.shape[2] // 2).float()])
    X, Y = torch.meshgrid(grid[0], grid[1])
    grid = torch.stack([X.flatten(), Y.flatten()])
    
    freq = torch.fft.fftfreq(img.shape[2], 1)
    pos = grid[0].reshape(*img.shape[1:])
    
    # First Step: Shear parallel to x-axis
    rimg = torch.fft.ifft(torch.fft.fft(img) * \
                          (2j * math.pi * t * freq * pos).exp()).abs()
    # Second Step: Shear parallel to y-axis
    rimg = torch.fft.ifft(
        torch.fft.fft(rimg.swapaxes(1, 2)) * \
        (2j * math.pi * s * freq * pos).exp()).swapaxes(1, 2).abs()
    # Third Step: as 1st
    rimg = torch.fft.ifft(torch.fft.fft(rimg) * \
                          (2j * math.pi * t * freq * pos).exp()).abs()
    return rimg


def alight_center(vec, center_fractions):
    """
    Takes motion vector and alight transition between 
    zeroed center and motion waves
    """
    column_num = vec.shape[0]
    center = column_num // 2
    central_columns = int(column_num * center_fractions)
    
    left_diff = vec[(center - central_columns // 2) - 1]
    right_diff = vec[center + central_columns // 2]

    vec = torch.cat((vec[0 : center - central_columns // 2] - left_diff,
                     vec[center - central_columns // 2 : center + \
                         central_columns // 2],
                     vec[center + central_columns // 2 :] - right_diff))
    return vec


def shift_vec_rand_tsavgol(column_num, amplitude,
                                  center_fractions=0.08, motion_num=8):
    """
    Sample motion vector from random normal pdf
    then smooth it with Savitzky–Golay filter
    x_y_max: ~ to max or min value of the sampled vector
    motion_num: too hard to control
    """
    
    x_shift = tsavgol(torch.randn(size=(1, column_num))[0], 20)
    y_shift = tsavgol(torch.randn(size=(1, column_num))[0], 26)
    
    x_shift = alight_center(normalize(x_shift) * amplitude, center_fractions)
    y_shift = alight_center(normalize(y_shift) * amplitude, center_fractions)
    shift_vector = torch.stack([x_shift, y_shift]) 
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)   
    shift_vector[:, center - central_columns // 2: center + \
                 central_columns // 2] = 0.
    return shift_vector

    
def rot_vec_rand_tsavgol(column_num, amplitude, center_fractions=0.08,
                                wave_num=2):
    rot_vector = tsavgol(torch.randn(size=(1, column_num))[0], 20)
    rot_vector = alight_center(normalize(rot_vector) * amplitude, 
                               center_fractions)
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)   
    rot_vector[center - central_columns // 2: center + \
               central_columns // 2] = 0.
    return rot_vector


def shift_vec_harmonic(column_num, amplitude, center_fractions=0.08,
                              motion_num=8):
    """
    Sample motion vector as a sum of sin and cos functions
    """
    shift_vector = torch.empty((2, column_num))    
    t = torch.linspace(0, motion_num * 2 * np.pi, column_num + 1)[:-1]
    
    a = torch.randint(1, 13, (1,))  # 12
    b = torch.randint(1, 6, (1,))  # 5
    c = np.around(random.uniform(0.2, 0.7), decimals=1) # 0.3
    d = np.around(random.uniform(0.1, 0.35), decimals=2)  # 0.17
    
    e = torch.randint(1, 6, (1,))
    f = torch.randint(1, 6, (1,))
    
    x_shift = -(t/(6*np.pi) + a).sin() + (t/(15*np.pi) + b).cos() + \
               ((t/(0.5 * np.pi * motion_num)).cos() * c) + \
               ((t/(0.3*np.pi)).cos() * d) + ((t/(0.7*np.pi)).sin() * 0.4)
    x_shift = alight_center(normalize(x_shift) * amplitude, center_fractions)
    
    y_shift = (t/(0.3 * np.pi) + e).cos() + \
              (t/(0.4 * np.pi * motion_num) + f).sin()
    y_shift = alight_center(normalize(y_shift) * amplitude, center_fractions)
    shift_vector = torch.stack([x_shift, y_shift]) 
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)   
    shift_vector[:, center - central_columns // 2: center + \
                 central_columns // 2] = 0.
    return shift_vector


def rot_vec_harmonic(column_num, amplitude, center_fractions=0.08,
                            wave_num=2):

    t = torch.linspace(0, wave_num * 2 * np.pi, column_num + 1)[:-1]
    
    a = torch.randint(1, 5, (1,))  #5
    b = torch.randint(1, 13, (1,))  # 0
    c = np.around(random.uniform(0.2, 0.6), decimals=1)  # 0.2
    d = random.uniform(1, 4)  

    rot_vector = (t/(6 * np.pi) + a + d).sin() + ((t/(2 * np.pi) + b).sin()) \
                 - ((t / (0.3 * np.pi)).sin() * c)
    rot_vector = alight_center(normalize(rot_vector) * amplitude,
                               center_fractions)
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)   
    rot_vector[center - central_columns // 2 : center \
               + central_columns // 2] = 0.
    return rot_vector


def shift_vec_periodic(column_num, amplitude, center_fractions=0.08,
                              motion_num=8):

    motion_num = torch.randint(motion_num-2, motion_num+3, (1,)).item()
    shift_vector = torch.empty((2, column_num))    
    t = torch.linspace(0, motion_num * 2 * np.pi, column_num + 1)[:-1]
    
    a = random.uniform(0.1, 3.0)
    b = random.uniform(1.0, 3.0)
    c = [-1,1][random.randrange(2)]
    d = [-1,1][random.randrange(2)]
    x_shift = c * np.cos(t + a)
    y_shift = d * np.cos(t * random.uniform(0.3, 1.0) + b)
    
    x_shift = alight_center(normalize(x_shift) * amplitude, center_fractions)
    y_shift = alight_center(normalize(y_shift) * amplitude, center_fractions)
    shift_vector = torch.stack([x_shift, y_shift]) 
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)   
    shift_vector[:, center - central_columns // 2: center + \
                 central_columns // 2] = 0.
    return shift_vector


def rot_vec_periodic(column_num, amplitude, center_fractions=0.08,
                            wave_num=2): 

    wave_num = torch.randint(wave_num-2, wave_num+3, (1,))
    t = np.linspace(0, wave_num * 2 * np.pi, column_num + 1)[:-1]
    
    a = random.uniform(0.1, 3.0)
    b = [-1,1][random.randrange(2)]

    rot_vector = torch.from_numpy(b * np.cos(t + a))
    rot_vector = alight_center(normalize(rot_vector) * amplitude,
                               center_fractions)
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)
    rot_vector[center - central_columns // 2: center + \
               central_columns // 2] = 0.
    return rot_vector


def sample_rot_vector(motion, column_num, theta, center_fractions=0.08,
                      wave_num=2):
    if motion == 'randomize_harmonic':
        rot_vector = rot_vec_harmonic(column_num, theta,
                                             center_fractions=0.08,
                                             wave_num=6)
    elif motion == 'randomize_periodic':
        rot_vector = rot_vec_periodic(column_num, theta,
                                             center_fractions=0.08,
                                             wave_num=6)
    elif motion == 'randomize_random':
        rot_vector = rot_vec_rand_tsavgol(column_num, theta,
                                             center_fractions=0.08,
                                             wave_num=6)
    else:
        raise ValueError('Incorrect motion type')
    return rot_vector.deg2rad()


def sample_shift_vector(motion, column_num, x_y_max, center_fractions=0.08,
                        motion_num=8):
    if motion == 'randomize_harmonic':
        shift_vector = shift_vec_harmonic(column_num, x_y_max,
                                          center_fractions=0.08,
                                          motion_num=8)
    elif motion == 'randomize_periodic':
        shift_vector = shift_vec_periodic(column_num, x_y_max,
                                          center_fractions=0.08,
                                          motion_num=8) 
    elif motion == 'randomize_random':
        shift_vector = shift_vec_rand_tsavgol(column_num, x_y_max,
                                              center_fractions=0.08,
                                              motion_num=8) 
    else:
        raise ValueError('Incorrect motion type')
    return shift_vector


class KTransform:
    """
    Do transform on k-space tensor with shape (N, H, W)
    """
    def __init__(self):
        pass

    def __call__(self, k_space: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    
class TranslationTransform2D(KTransform):

    def __init__(self, x_y_shift=0.0, motion_num=6, motion='harmonic'):
        super(TranslationTransform2D, self).__init__()
        self.x_y_shift = x_y_shift
        self.motion_num = motion_num
        self.motion = motion
        
    def __call__(self, k_space: torch.Tensor, center_fractions):
        shift_vector = sample_shift_vector(self.motion, k_space.shape[-1],
                                           self.x_y_shift,
                                           center_fractions=center_fractions,
                                           motion_num=self.motion_num)
        x_shift = shift_vector[0]
        y_shift = shift_vector[1]
        x_shape, y_shape = k_space.shape[-2:]

        phase_shift = -2 * math.pi * (
            x_shift * torch.linspace(0, 1, x_shape)[None, :, None] +
            y_shift * torch.linspace(0, 1, y_shape)[None, None, :])
        new_k_space = k_space.abs() * (1j * (k_space.angle() + \
                                             phase_shift)).exp()
        return new_k_space, shift_vector
    

class NUFFT_RotationTransform2D(KTransform):
    """Rotate each column of k-space via NU-FFT"""

    def __init__(self, theta=0.0, wave_num=2, center_fractions=0.08,
                 motion='harmonic'):
        super(NUFFT_RotationTransform2D, self).__init__()
        self.theta = theta  # in degrees
        self.wave_num = wave_num
        self.center_fractions = center_fractions
        self.motion = motion

    def __call__(self, k_space: torch.Tensor) -> torch.Tensor:
        k_space = k_space[0]
        rot_vector = sample_rot_vector(self.motion, k_space.shape[-1],
                                       self.theta,
                                       center_fractions=self.center_fractions,
                                       wave_num=self.wave_num).float()
        rot_matrices = get_rot_mat_nufft(rot_vector)
        grid = torch.stack([
            arr.flatten() for arr in torch.meshgrid(
                torch.arange(-k_space.shape[0]//2, k_space.shape[0]//2).float(),
                torch.arange(-k_space.shape[1]//2, k_space.shape[1]//2).float(),
                indexing='ij')])#.cuda()

        grid = (rot_matrices @ \
                grid.reshape(2, k_space.shape[-1], 
                             k_space.shape[-1]).movedim(1, 0)).movedim(0, 1).reshape(2, -1)

        img = nufft.nufft_adjoint(k_space, grid.T, device='cpu', oversamp=5, 
                                  out_shape=[1, 1, *k_space.shape])[0, 0]
        return Ft(img), rot_vector
    
    
class RotationTransform2D(KTransform):
    """Rotate each column of k-space via Shear Transform"""

    def __init__(self, theta=0.0, wave_num=2, center_fractions=0.08):
        super(RotationTransform2D, self).__init__()
        self.theta = theta  # in degrees
        self.wave_num = wave_num
        self.center_fractions = center_fractions

    def __call__(self, k_space: torch.Tensor) -> torch.Tensor:

        rot_vector = sample_rot_vector(k_space.shape[-1], self.theta,
                                       center_fractions=self.center_fractions,
                                       wave_num=self.wave_num)
        img = pt_kspace2spatial(k_space).abs()

        new_k_space = torch.empty((1, k_space.shape[-2], k_space.shape[-1]),
                                  dtype=torch.complex128)
        for col_idx, theta in enumerate(rot_vector):
            rot_k_space = pt_spatial2kspace(_shear_rot_img(img, theta))
            
            new_k_space[:, :, col_idx] = rot_k_space[:, :, col_idx]

        return new_k_space, rot_vector 


class RandomTranslationTransform(KTransform):

    def __init__(self, xy_max: float, motion_num: float,
                 center_fractions: float, motion: str):
        super(RandomTranslationTransform, self).__init__()
        self.xy_max = xy_max
        self.motion_num = motion_num
        self.center_fractions = center_fractions
        self.motion = motion
        
        self.translate = TranslationTransform2D()

    def resample(self):
        self.translate.x_y_shift = self.xy_max
        self.translate.motion_num = self.motion_num
        self.translate.motion = self.motion

    def __call__(self, k_space: torch.Tensor) -> torch.Tensor:
        self.resample()
        return self.translate(k_space, center_fractions=self.center_fractions)


class RandomRotationTransform(KTransform):
    """Class that samples randomness into the rotation motion"""

    def __init__(self, wave_num: int, theta_max: float,
                 center_fractions: float, motion: str, 
                 positive_angles_only=False):
        super(RandomRotationTransform, self).__init__()
        self.theta_max = theta_max
        self.wave_num = wave_num
        self.positive_angles_only = positive_angles_only
        self.center_fractions = center_fractions
        self.motion = motion
        
        self.rotate = NUFFT_RotationTransform2D()

    def resample(self):
        self.rotate.theta = self.theta_max
        self.rotate.wave_num = self.wave_num
        self.rotate.center_fractions = self.center_fractions
        self.rotate.motion = self.motion
    
    def __call__(self, k_space):
        self.resample()
        return self.rotate(k_space)


class RandomMotionTransform(KTransform):

    def __init__(self, xy_max: float, theta_max: float, num_motions: int,
                 wave_num: int, center_fractions: float, motion_type: str,
                 noise_lvl: float):
        super(RandomMotionTransform, self).__init__()
        self.xy_max = xy_max
        self.theta_max = theta_max
        self.noise_lvl = noise_lvl
        self.T = RandomTranslationTransform(xy_max, num_motions,
                                            center_fractions,
                                            motion_type)
        self.R = RandomRotationTransform(wave_num, theta_max,
                                         center_fractions,
                                         motion_type,
                                         positive_angles_only=True)

    def __call__(self, k_space: torch.Tensor) -> torch.Tensor:

        rot_vector = torch.zeros((320))
        k_space, rot_vector = self.R(k_space)

        shift_vector = torch.zeros((2, 320))
        k_space, shift_vector = self.T(k_space)
        
        if self.noise_lvl != 0:
            k_space = add_noise(k_space, self.noise_lvl)
        return k_space, rot_vector, shift_vector
