import torch
import torch.nn as nn
from math import factorial
import numpy as np
import torch.nn.functional as F

eps = 1e-8


def get_kernel(name):
    """Return kernels for grad along x-axis.
    Transpose to have kernel for grad along to y-axis"""
    
    name = 'scharr'  # TODO: for now kernel type is hardcoded
    
    if name == 'sobel':
        kernel = torch.tensor([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])
    elif name == 'scharr':
        kernel = torch.tensor([[3, 0, -3],
                               [10, 0, -10],
                               [3, 0, -3]])
    else:
        kernel = torch.tensor([[1],
                               [-1]])
    return kernel.float()


def image_gradient_x(img, kernel_name='default'):
    kernel_vert = get_kernel(kernel_name).cuda()
    
    conv1=nn.Conv2d(1, 1, kernel_size=kernel_vert.shape, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(kernel_vert[None, None])
    G_x = conv1(img)
    
    if G_x.shape != img.shape:
        dx, dy = G_x.shape[-2] - img.shape[-2], G_x.shape[-1] - img.shape[-1]
        G_x = G_x[:, :, dx // 2 : dx // 2 + img.shape[-2],
                        dy // 2 : dy // 2 + img.shape[-2]]
    return G_x


def image_gradient_y(img, kernel_name='default'):
    kernel_horiz = get_kernel(kernel_name).T.cuda()
    
    conv1=nn.Conv2d(1, 1, kernel_size=kernel_horiz.shape, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(kernel_horiz[None, None])
    G_y = conv1(img)
    
    if G_y.shape != img.shape:
        dx, dy = G_y.shape[-2] - img.shape[-2], G_y.shape[-1] - img.shape[-1]
        G_y = G_y[:, :, dx // 2 : dx // 2 + img.shape[-2],
                        dy // 2 : dy // 2 + img.shape[-2]]
    return G_y


def image_gradient(img, kernel_name='default'):
    G_x = image_gradient_x(img, kernel_name)
    G_y = image_gradient_y(img, kernel_name)
        
    G=torch.sqrt(torch.pow(G_x, 2)+ torch.pow(G_y, 2))
    return G


# Entropy Function
v_func = lambda u : (u * u.conj() + eps**2).sqrt() / (u.T @ u + eps**2).sqrt()
entropy = lambda u : -v_func(u).T @ v_func(u).log()


# Means 
def grad_mean_X(img):
    G_x = image_gradient_x(img)
    G_x = G_x.abs()
    return G_x.mean()


def grad_mean_Y(img):
    G_x = image_gradient_y(img)
    G_x = G_x.abs()
    return G_x.mean()


def grad_mean_BOTH(img):
    G_x = image_gradient(img)
    G_x = G_x.abs()
    return G_x.mean()


def img_mean(img):
    return img.mean()


# Entropy of Normalized Gradient
def gradient_entropy(img):
    G_x = image_gradient_x(img)
    G_x = G_x.abs()
    G_x = G_x / torch.sum(G_x)
    
    return entropy(G_x[0][0].view(-1))

def gradient_entropy_Y(img): 
    G_x = image_gradient_y(img)
    
    G_x = G_x.abs()
    G_x = G_x / torch.sum(G_x)
    
    return entropy(G_x[0][0].view(-1))

def gradient_entropy_BOTH(img):
    G_x = image_gradient(img)
    
    G_x = G_x.abs()
    G_x = G_x / torch.sum(G_x)
    
    return entropy(G_x[0][0].view(-1))


# Normalized Gradient Squared
def norm_grad_sqr(img):  
    G_x = image_gradient_x(img) + eps**2
    G_x = G_x.abs()
    G_x = torch.square(G_x / torch.sum(G_x))
    
    return torch.sum(G_x[0][0])


def norm_grad_sqr_Y(img):
    G_x = image_gradient_y(img) + eps**2
    G_x = G_x.abs()
    G_x = torch.square(G_x / torch.sum(G_x))
    
    return torch.sum(G_x[0][0])


def norm_grad_sqr_BOTH(img):
    G_x = image_gradient(img) + eps**2
    G_x = G_x.abs()
    G_x = torch.square(G_x / torch.sum(G_x))
    
    return torch.sum(G_x[0][0])


# Smoothing Algorithms

def tsavgol(signal, kernel=6, stride=2, N=3):
    assert len(signal.shape) == 1
    if kernel < N + 2:
        raise TypeError("kernel size {} is too small for the polynomials order {}".format(kernel, N))
        
    patches = torch.ones_like(signal).unfold(0, kernel, stride)  # below fix hardcoded (320, 1)
    m = F.fold(patches.permute(1,0)[None], output_size=(320, 1), kernel_size=(kernel, 1), stride=(stride, 1))[0, 0, :, 0]
    patches = signal.unfold(0, kernel, stride)
    new_patches = torch.zeros_like(patches)
    for i in range(patches.shape[0]):
        x = torch.linspace(-1, 1, patches[i].shape[0]).to(signal.device)
        V = torch.vander(x, N=N).to(signal.device)
        p = V.pinverse() @ patches[i].view(-1, 1)
        new_patches[i] = (V @ p).flatten()
    return F.fold(new_patches.permute(1,0)[None], output_size=(signal.shape[0], 1),
                  kernel_size=(kernel, 1), stride=(stride, 1))[0, 0, :, 0] / m


def savitzky_golay(rot_vector, window_size=11, order=3, deriv=0, rate=1):
    y = rot_vector
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
        
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    
    # precompute coefficients
    bb = torch.vander(torch.arange(-half_window, half_window+1), N=order+1, increasing=True).float().to(rot_vector.device)

    kernel = (torch.linalg.pinv(bb)[deriv] * rate** deriv * factorial(deriv)).float().to(rot_vector.device)
    
    # pad the signal at the extremes with values taken from the signal itself
    firstvals = rot_vector[0] - torch.abs(torch.flip(rot_vector[1:half_window+1], (0, )) - rot_vector[0])
    lastvals = rot_vector[-1] + torch.abs(torch.flip(rot_vector[-half_window-1:-1], (0, )) - rot_vector[-1])
    rot_vector = torch.cat([firstvals, rot_vector, lastvals])
    
    conv=nn.Conv1d(1, 1, kernel_size=kernel.shape, stride=1, padding=0, bias=False)
    conv.weight=nn.Parameter(kernel[None, None])
    
    return conv(rot_vector[None, None])


def moving_average(rot_vector, window_size=11):
    """
    rot_vector: signal of shape [1, 1, N]
    """
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
        
    window = (torch.ones((window_size)) / window_size).float().to(rot_vector.device)
    
    conv=nn.Conv1d(1, 1, kernel_size=window_size, stride=1, padding=window_size//2, bias=False)
    conv.weight=nn.Parameter(window[None, None])
    
    return conv(rot_vector)


def FFt_smoothing(rot_vector):
    # TODO with some low pass filter
    return rot_vector


def spline_approximation(rot_vector):
    # TODO: b-spline for example (best fits for medical images)
    return rot_vector


def median_filter(rot_vector, window_size=11):
    """
    rot_vector: signal of shape [N, 1]
    """
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
        
    rot_vector_shape = rot_vector.shape[0]
    half_window = (window_size -1) // 2
    median_vector = torch.zeros_like(rot_vector).to(rot_vector.device)
    
    firstvals = rot_vector[0] - torch.abs(torch.flip(rot_vector[1:half_window+1], (0, )) - rot_vector[0])
    lastvals = rot_vector[-1] + torch.abs(torch.flip(rot_vector[-half_window-1:-1], (0, )) - rot_vector[-1])
    rot_vector = torch.cat([firstvals, rot_vector, lastvals])

    for elem in range(rot_vector_shape):
        median_vector[elem] = torch.median(rot_vector[elem : elem + window_size])
    
    return median_vector


def spike_killer_3000(rot_vector, threshold): # TODO mean -> interpolation, increase window to check
    rot_shift_delta = torch.abs(rot_vector - torch.roll(rot_vector, 1, 0))
    rot_shift_delta[0] = 0.

    new_rot_vector = torch.where(rot_shift_delta > threshold, rot_vector.mean(), rot_vector)
    return new_rot_vector
