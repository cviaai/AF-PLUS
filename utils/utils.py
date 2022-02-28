import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from k_space_reconstruction.utils.metrics import psnr, ssim
import piq


def sample_noise(motion_vector, noise_lvl):
    n = torch.randn(size=motion_vector.shape) * motion_vector.mean()
    return n * noise_lvl


def add_noise(ks, noise_level, noise_in_range=False):
    if noise_in_range:
        noise_level = torch.randint(low=0, high=noise_level + 20, size=(1, ))
    ks = ks + torch.randn(size=ks.shape) * ks.mean() * noise_level
    return ks


def normalize(x):
    x1 = x - x.min()
    return x1 / x1.max()
    
    
def calc_metrics(y_pred: torch.Tensor, y_gt: torch.Tensor):

    metrics_dict = {}
    
    metrics_dict['psnr'] = psnr(y_pred, y_gt).item()
    metrics_dict['ssim'] = ssim(y_pred, y_gt).item()
    metrics_dict['l1_loss'] = F.l1_loss(y_pred, y_gt).item()
    metrics_dict['ms_ssim'] = piq.multi_scale_ssim(normalize(y_pred), 
                                                   normalize(y_gt),
                                                   data_range=1.).item()
    metrics_dict['vif_p'] = piq.vif_p(normalize(y_pred), normalize(y_gt), 
                                      data_range=1.).item()
    return metrics_dict


def plot_motion_vector(vec, motion_type):
    size = vec.shape[-1]
    
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()

    x_axis = np.linspace(-3.14, 3.14, 320)
    if motion_type == 'translation':
        ax.plot(x_axis, vec[0], label='x', color='green')
        ax.plot(x_axis, vec[1], label='y', color='blue')
        ax.set_ylabel('Shift Value, (pixel)')
        ax.legend()
    elif motion_type == 'rotation':
        ax.plot(x_axis, vec[0], label='x', color='green')
        ax.set_ylabel('Rotation Value, (degree)')
    else:
        raise TypeError('Incorrect motion vector type!')

    ax.set_xlabel('K-space Phase, (radians)')
    # Set values for x-axis
    new_tick_locations = np.linspace(-3.14, 3.14, 15)
    new_tick_labels = np.round(np.linspace(-3.14, 3.14, 15), 2)
    ax.set_xticks(new_tick_locations)
    ax.set_xticklabels(new_tick_labels)
    
    # Set values for 2nd x-axis
    new_tick_labels = np.linspace(0, size, 15).astype(np.int32)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(new_tick_labels)
    ax2.set_xlabel('K-space column, (unit)')

    plt.title('Motion Vector for {}'.format(motion_type))
    
    plt.tight_layout()
    plt.grid()
    plt.show()