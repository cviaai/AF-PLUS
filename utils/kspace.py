import math
import random
import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import contextlib
from typing import Optional, Sequence, Tuple, Union, List
import matplotlib.pyplot as plt
from k_space_reconstruction.utils.optimization_losses import tsavgol

import sys
sys.path.append('./pytorch_nufft')
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
    rimg = torch.fft.ifft(torch.fft.fft(img) * (2j * math.pi * t * freq * pos).exp()).abs()
    # Second Step: Shear parallel to y-axis
    rimg = torch.fft.ifft(
        torch.fft.fft(rimg.swapaxes(1, 2)) * (2j * math.pi * s * freq * pos).exp()).swapaxes(1, 2).abs()
    # Third Step: as 1st
    rimg = torch.fft.ifft(torch.fft.fft(rimg) * (2j * math.pi * t * freq * pos).exp()).abs()

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
                     vec[center - central_columns // 2 : center + central_columns // 2],
                     vec[center + central_columns // 2 :] - right_diff))
    return vec


def sample_shift_vec_rand_tsavgol(column_num, amplitude, center_fractions=0.08,
                                  motion_num=8):
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
    shift_vector[:, center - central_columns // 2: center + central_columns // 2] = 0.
    
    return shift_vector

    
def sample_rot_vec_rand_tsavgol(column_num, amplitude, center_fractions=0.08,
                                wave_num=2):

    rot_vector = tsavgol(torch.randn(size=(1, column_num))[0], 20)
    rot_vector = alight_center(normalize(rot_vector) * amplitude, center_fractions)
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)   
    rot_vector[center - central_columns // 2: center + central_columns // 2] = 0.
    
    return rot_vector


def sample_shift_vec_harmonic(column_num, amplitude, center_fractions=0.08,
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
    shift_vector[:, center - central_columns // 2: center + central_columns // 2] = 0.

    return shift_vector


def sample_rot_vec_harmonic(column_num, amplitude, center_fractions=0.08,
                            wave_num=2):

    t = torch.linspace(0, wave_num * 2 * np.pi, column_num + 1)[:-1]
    
    a = torch.randint(1, 5, (1,))  #5
    b = torch.randint(1, 13, (1,))  # 0
    c = np.around(random.uniform(0.2, 0.6), decimals=1)  # 0.2
    d = random.uniform(1, 4)  

    rot_vector = (t/(6*np.pi) + a + d).sin() + ((t/(2*np.pi) + b).sin()) \
                 - ((t/(0.3*np.pi)).sin() * c)
    rot_vector = alight_center(normalize(rot_vector) * amplitude, center_fractions)
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)   
    rot_vector[center - central_columns // 2 : center + central_columns // 2] = 0.
    
    return rot_vector


def sample_shift_vec_periodic(column_num, amplitude, center_fractions=0.08,
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
    shift_vector[:, center - central_columns // 2: center + central_columns // 2] = 0.
    
    return shift_vector


def sample_rot_vec_periodic(column_num, amplitude, center_fractions=0.08,
                            wave_num=2): 

    wave_num = torch.randint(wave_num-2, wave_num+3, (1,))
    t = np.linspace(0, wave_num * 2 * np.pi, column_num + 1)[:-1]
    
    a = random.uniform(0.1, 3.0)
    b = [-1,1][random.randrange(2)]

    rot_vector = torch.from_numpy(b * np.cos(t + a))
    rot_vector = alight_center(normalize(rot_vector) * amplitude, center_fractions)
    
    center = column_num // 2
    central_columns = int(column_num * center_fractions)
    rot_vector[center - central_columns // 2: center + central_columns // 2] = 0.
    return rot_vector


#------------------------------------------------------------------------------
def sample_rot_vector(motion, column_num, theta, center_fractions=0.08,
                      wave_num=2):
    if motion == 'harmonic':
        name = 'motion_vectors/rot_vec_harmonic_1.pt'
        rot_vector = torch.load(name)
        
    elif motion == 'periodic':
        name = 'motion_vectors/rotation_vector_05_degree.pt'
        rot_vector = torch.load(name) 
        
    elif motion == 'random':
        name = 'motion_vectors/rot_vec_random_3_24.pt'
        rot_vector = torch.load(name)
        
    elif motion == 'randomize_harmonic':
        rot_vector = sample_rot_vec_harmonic(column_num, theta,
                                             center_fractions=0.08,
                                             wave_num=6)
    elif motion == 'randomize_periodic':
        rot_vector = sample_rot_vec_periodic(column_num, theta,
                                             center_fractions=0.08,
                                             wave_num=6)
    elif motion == 'randomize_random':
        rot_vector = sample_rot_vec_rand_tsavgol(column_num, theta,
                                             center_fractions=0.08,
                                             wave_num=6)
    else:
        raise ValueError('Incorrect motion type')
    return rot_vector.deg2rad()


#------------------------------------------------------------------------------
def sample_shift_vector(motion, column_num, x_y_max, center_fractions=0.08,
                        motion_num=8):
    if motion == 'harmonic':
        name = 'motion_vectors/shift_vec_harmonic_1.pt'
        shift_vector = torch.load(name)
        
    elif motion == 'periodic':
        name = 'motion_vectors/shift_vector_1_pixels.pt'
        shift_vector = torch.load(name) * 0.5
        
    elif motion == 'random':
        name = 'motion_vectors/shift_vec_random_3_24.pt'
        shift_vector = torch.load(name) * 0.6

    elif motion == 'randomize_harmonic':
        shift_vector = sample_shift_vec_harmonic(column_num, x_y_max,
                                                 center_fractions=0.08,
                                                 motion_num=8)
    elif motion == 'randomize_periodic':
        shift_vector = sample_shift_vec_periodic(column_num, x_y_max,
                                                 center_fractions=0.08,
                                                 motion_num=8) 
    elif motion == 'randomize_random':
        shift_vector = sample_shift_vec_rand_tsavgol(column_num, x_y_max,
                                                 center_fractions=0.08,
                                                 motion_num=8) 
    else:
        raise ValueError('Incorrect motion type')
    return shift_vector
#------------------------------------------------------------------------------


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
        
    def __call__(self, k_space: torch.Tensor, center_fractions) -> torch.Tensor:
        shift_vector = sample_shift_vector(self.motion, k_space.shape[-1], self.x_y_shift,
                                           center_fractions=center_fractions,
                                           motion_num=self.motion_num)
        x_shift = shift_vector[0]
        y_shift = shift_vector[1]
        x_shape, y_shape = k_space.shape[-2:]

        phase_shift = -2 * math.pi * (
            x_shift * torch.linspace(0, 1, x_shape)[None, :, None] +
            y_shift * torch.linspace(0, 1, y_shape)[None, None, :]
        )
        new_k_space = k_space.abs() * (1j * (k_space.angle() + phase_shift)).exp()
        return new_k_space, shift_vector
    
    
class NUFFT_RotationTransform2D(KTransform):
    """Rotate each column of k-space via NU-FFT"""

    def __init__(self, theta=0.0, wave_num=2, center_fractions=0.08, motion='harmonic'):
        super(NUFFT_RotationTransform2D, self).__init__()
        self.theta = theta  # in degrees
        self.wave_num = wave_num
        self.center_fractions = center_fractions
        self.motion = motion

    def __call__(self, k_space: torch.Tensor) -> torch.Tensor:
        k_space = k_space[0]
        rot_vector = sample_rot_vector(self.motion, k_space.shape[-1], self.theta,
                                       center_fractions=self.center_fractions,
                                       wave_num=self.wave_num).float()
        grid = torch.stack([
            arr.flatten() for arr in torch.meshgrid(
                torch.arange(-k_space.shape[0]//2, k_space.shape[0]//2).float(),
                torch.arange(-k_space.shape[1]//2, k_space.shape[1]//2).float(),
                indexing='ij')])

        for i in range(k_space.shape[0]):
            R = get_rot_mat(rot_vector[i])
            grid[:, k_space.shape[1] * i : k_space.shape[1] * (i+1)] = R @ grid[:, k_space.shape[1] * i : k_space.shape[1] * (i+1)]

        img = nufft.nufft_adjoint(k_space, grid.T, device='cpu', oversamp=5, out_shape=[1, 1, *k_space.shape])[0]
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

        new_k_space = torch.empty((1, k_space.shape[-2], k_space.shape[-1]), dtype=torch.complex128)
        for col_idx, theta in enumerate(rot_vector):
            rot_k_space = pt_spatial2kspace(_shear_rot_img(img, theta)) #complex128, [1, 320, 320]
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
#         self.translate.x_shift = ((torch.rand(1) - 0.5) * 2 * self.xy_max).item()  # random sampling
        self.translate.motion = self.motion

    def __call__(self, k_space: torch.Tensor) -> torch.Tensor:
        self.resample()
        return self.translate(k_space, center_fractions=self.center_fractions)


class RandomRotationTransform(KTransform):
    """Class that samples randomness into the rotation motion"""

    def __init__(self, wave_num: int, theta_max: float,
                 center_fractions: float, motion: str, positive_angles_only=False):
        super(RandomRotationTransform, self).__init__()
        self.theta_max = theta_max
        self.wave_num = wave_num
        self.positive_angles_only = positive_angles_only
        self.center_fractions = center_fractions
        self.motion = motion
        
        self.rotate = NUFFT_RotationTransform2D()

    def resample(self):
#         rand_theta = (torch.rand(1) - 0.5) * 2 * self.theta_max  # TMP
#         if self.positive_angles_only:
#             rand_theta = rand_theta.abs()
#         self.rotate.theta = rand_theta.item() # TODO: temporary hardcode this
#         self.rotate.wave_num = torch.randint(low=0, high=3, size=(1,)).item() + self.wave_num

        self.rotate.theta = self.theta_max
        self.rotate.wave_num = self.wave_num
        self.rotate.center_fractions = self.center_fractions
        self.rotate.motion = self.motion
    
    def __call__(self, k_space):
        self.resample()
        return self.rotate(k_space)


class RandomMotionTransform(KTransform):

    def __init__(self, xy_max: float, theta_max: float, num_motions: int, wave_num: int,
                 center_fractions: float, motion_type: str, noise_lvl: float):
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
        
#         plt.imshow(IFt(k_space[0]).abs())
#         plt.show()

        rot_vector = torch.zeros((320))
        k_space, rot_vector = self.R(k_space)

        shift_vector = torch.zeros((2, 320))
        k_space, shift_vector = self.T(k_space)
        
        if self.noise_lvl != 0:
            k_space = add_noise(k_space, self.noise_lvl)
            
        return k_space, rot_vector, shift_vector

    
class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.
    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).
    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        """
        Create the mask.
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.
        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 2:
            raise ValueError("Shape should have 2 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-1]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
            )
            mask = self.rng.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad:pad + num_low_freqs] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-1] = num_cols
            mask = mask.reshape(*mask_shape).astype(np.float32)

        return mask


class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding tovlow-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.
    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        """
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.
        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 2:
            raise ValueError("Shape should have 2 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-1]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols
            )
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-1] = num_cols
            mask = mask.reshape(*mask_shape).astype(np.float32)

        return mask


def apply_mask(
    data: np.ndarray,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.
    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = data.shape
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask