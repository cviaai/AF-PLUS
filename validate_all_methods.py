import math
import argparse
import random
import numpy as np
import pylab as plt
import torch
import torch.nn.functional as F
import h5py
import piq
import pandas as pd

from utils.unet import Unet
from tqdm import tqdm
import skimage.data

from config import PATH

from utils.fastmri import FastMRITransform, DemotionFastMRIh5Dataset
from utils.kspace import RandomMotionTransform
from utils.unet import Unet
from utils.utils import l1_loss
from utils.utils import t2i, normalize, psnr, ssim

import sys
sys.path.append(PATH.NUFFT_PATH)
import nufft
from torch.fft import fftshift, ifftshift, fftn, ifftn

from oct2py import octave
octave.addpath(PATH.GRADMC_PATH + 'code/')
octave.addpath(PATH.GRADMC_PATH + 'code/@matFastFFTmotion/private')
octave.addpath(PATH.GRADMC_PATH)
octave.warning('off', 'all')
    
# All functions are validated by me
from autofocusing_plus_train import load_val_dataset
from autofocusing_plus_train import simult_de_motion, R_differentiable


Ft = lambda x : fftshift(fftn(ifftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
IFt = lambda x : ifftshift(ifftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

    
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


def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('motion_type', type=str, 
                        help='motion type that is used')
    parser.add_argument('val_size', type=int, default=53, 
                        help='size of validation dataset')
    args = parser.parse_args() 
    return args


def just_unet(ks, model):
    unet = Unet(1, 1, 32, 6, batchnorm=torch.nn.InstanceNorm2d, init_type='none').cuda()
    unet.load_state_dict(torch.load(PATH.MODEL_UNET_PATH))
    unet.eval()
    img = IFt(ks).abs().cuda()
    y_pred = unet(img[None, None].cuda())
    return y_pred[0][0]


def af(ks):
    beta1, beta2 = 0.89, 0.8999
    ps = ks.shape[-1]
    ps_cf = int((ps // 2) * 0.08)
    zero_middle = torch.ones((ps)).cuda()
    zero_middle[ps // 2 - ps_cf : ps // 2 + ps_cf] = 0.
    img = IFt(ks).abs()

    # Translation Params
    x_shifts = torch.zeros(ps)
    y_shifts = torch.zeros(ps)
    x_shifts = torch.nn.Parameter(data=x_shifts.cuda(), requires_grad=True)
    y_shifts = torch.nn.Parameter(data=y_shifts.cuda(), requires_grad=True)
    x_moment1 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
    x_moment2 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
    y_moment1 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
    y_moment2 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
    # Rotation Params
    rot_vector = torch.zeros(ks.shape[-1]).cuda()
    rot_vector = torch.nn.Parameter(data=rot_vector.cuda(), requires_grad=True)
    rot_moment1 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
    rot_moment2 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
    
    for _ in range(80):
        rot_vector = rot_vector * zero_middle
        x_shifts = x_shifts * zero_middle
        y_shifts = y_shifts * zero_middle
        # Translation
        phase_shift = -2 * math.pi * (
            x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
            y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
        yp_ks = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
        # Rotation
        new_k_space = R_differentiable(yp_ks, rot_vector)
        yp_img = IFt(new_k_space).abs()

        loss_net = (yp_img[None] * 1e4).mean()
        x_grad, y_grad, rot_grad = torch.autograd.grad(loss_net, [x_shifts, y_shifts, rot_vector],
                                                       create_graph=False)
        x_grad, y_grad, rot_grad = x_grad * 1e-4, y_grad * 1e-4, rot_grad * 1e-4
        x_moment1 = beta1 * x_moment1 + (1. - beta1) * x_grad
        x_moment2 = beta2 * x_moment2 + (1. - beta2) * x_grad * x_grad + 1e-24
        y_moment1 = beta1 * y_moment1 + (1. - beta1) * y_grad
        y_moment2 = beta2 * y_moment2 + (1. - beta2) * y_grad * y_grad + 1e-24
        x_shifts = x_shifts - 3e-4 * x_moment1 * x_moment2.rsqrt()
        y_shifts = y_shifts - 3e-4 * y_moment1 * y_moment2.rsqrt()
        rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
        rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
        rot_vector = rot_vector - 3e-4  * rot_moment1 * rot_moment2.rsqrt()
 
    rot_vector = rot_vector * zero_middle
    x_shifts = x_shifts * zero_middle
    y_shifts = y_shifts * zero_middle
    # Translation
    phase_shift = -2 * math.pi * (
        x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
        y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
    yp_ks = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
    # Rotation
    new_k_space = R_differentiable(yp_ks, rot_vector)
    return new_k_space


def grad_mc(ks):
    return torch.from_numpy(octave.myGradMC(ks.numpy())).abs()

    
def af_unet(ks, model):
    
    unet = Unet(1, 1, 32, 6, batchnorm=torch.nn.InstanceNorm2d, init_type='none').cuda()
    unet.load_state_dict(torch.load(PATH.MODEL_AF_PATH))
    unet.eval()
    
    beta1, beta2 = 0.89, 0.8999
    ps = ks.shape[-1]
    ps_cf = int(ps // 2 * 0.08)
    zero_middle = torch.ones((ps)).cuda()
    zero_middle[ps // 2 - ps_cf : ps // 2 + ps_cf] = 0.
    img = IFt(ks).abs()

    x_shifts = torch.zeros(ps)
    y_shifts = torch.zeros(ps)
    x_shifts = torch.nn.Parameter(data=x_shifts.cuda(), requires_grad=True)
    y_shifts = torch.nn.Parameter(data=y_shifts.cuda(), requires_grad=True)
    x_moment1 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
    x_moment2 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
    y_moment1 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
    y_moment2 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
    rot_vector = torch.zeros(ps).cuda()
    rot_vector = torch.nn.Parameter(data=rot_vector.cuda(), requires_grad=True)
    rot_moment1 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
    rot_moment2 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)

    for _ in range(80):
        rot_vector = rot_vector * zero_middle
        x_shifts = x_shifts * zero_middle
        y_shifts = y_shifts * zero_middle

        # Translation
        phase_shift = -2 * math.pi * (
            x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
            y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
        new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + \
                                               phase_shift)).exp()
        # Rotation
        yp_ks = R_differentiable(new_k_space, rot_vector)
        yp_img = IFt(yp_ks).abs()

        loss_net = (yp_img[None, None] * 1e4 * unet(yp_img[None, None] * 1e4).sigmoid()).mean()
        x_grad, y_grad, rot_grad = torch.autograd.grad(loss_net,
                                                       [x_shifts, y_shifts, rot_vector], create_graph=False)
        x_grad, y_grad = x_grad * 1e-4, y_grad * 1e-4
        rot_grad = rot_grad * 1e-4
        x_moment1 = beta1 * x_moment1.detach() + (1. - beta1) * x_grad
        x_moment2 = beta2 * x_moment2.detach() + (1. - beta2) * x_grad * x_grad + 1e-24
        y_moment1 = beta1 * y_moment1.detach() + (1. - beta1) * y_grad
        y_moment2 = beta2 * y_moment2.detach() + (1. - beta2) * y_grad * y_grad + 1e-24
        rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
        rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
        x_shifts = x_shifts - 3e-4 * x_moment1 * x_moment2.rsqrt()
        y_shifts = y_shifts - 3e-4 * y_moment1 * y_moment2.rsqrt()
        rot_vector = rot_vector - 3e-4 * rot_moment1 * rot_moment2.rsqrt()

    rot_vector = rot_vector * zero_middle
    x_shifts = x_shifts * zero_middle
    y_shifts = y_shifts * zero_middle
    # Translation
    phase_shift = -2 * math.pi * (
        x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
        y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
    new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + \
                                           phase_shift)).exp()
    # Rotation
    yp_ks = R_differentiable(new_k_space, rot_vector)
    return yp_ks
 

def run_image_demotion(ks, gt_ks, model_af, model_nn):
    
    af_unet_restored = af_unet(ks, model_af)  # ks
    unet_restored = just_unet(ks, model_nn)  # img
    af_restored = af(ks)                     # ks
    gradmc_restored = grad_mc(ks)           # img
    return {'af_unet_ks': af_unet_restored,
            'af_ks': af_restored,
            'unet_img': unet_restored,
            'gradmc_img': gradmc_restored}


def load_val_dataset(motion_type, n_item):
        
#     val_data_path = PATH.VAL_PATH + '{}.h5'.format(motion_type)
    val_data_path = PATH.VAL_PATH + PATH.VAL_NAME

    hf = h5py.File(val_data_path)
    val_dataset = []
    for f in tqdm(sorted(list(hf.keys())[:n_item])):
        batch = hf[f]
        ks = torch.from_numpy(batch[0])
        ks = torch.stack([ks.real, ks.imag]).to(torch.complex64)
        gt_ks = torch.from_numpy(batch[1])
        gt_ks = torch.stack([gt_ks.real, gt_ks.imag]).to(torch.complex64)
        d = {'k_space': ks,
             'target_k_space': gt_ks}
        val_dataset.append(d)  
    return val_dataset


if __name__ == '__main__': 
    args = parsing_args()

    random.seed(228)
    torch.manual_seed(228)
    torch.cuda.manual_seed(228)
    np.random.seed(228)

    csv_name = PATH.SAVING_PATH + 'validation_{}.csv'.format(args.motion_type)
    with open(csv_name, "w") as f:
        pass

    # Load dataset
    val_dataset = load_val_dataset(args.motion_type, args.val_size)

    af_metrics, unet_metrics = [], []
    af_unet_metrics, old_metric = [], []
    grad_metric = []
    
    for batch_idx in tqdm(range(0, len(val_dataset))):
        batch = val_dataset[batch_idx]
        gt_ks = batch['target_k_space']
        gt_ks = gt_ks[0] + 1j * gt_ks[1]
        ks = batch['k_space']
        ks = ks[0] + 1j * ks[1]

        restored_dict = run_image_demotion(ks, gt_ks, PATH.MODEL_AF_PATH, PATH.MODEL_UNET_PATH)

        old_metric.append(calc_metrics(IFt(ks).abs()[None, None].cpu().detach(),
                                  IFt(gt_ks).abs()[None, None].cpu().detach()))
        af_metrics.append(calc_metrics(IFt(restored_dict['af_ks']).abs()[None, None].cpu().detach(),
                                  IFt(gt_ks).abs()[None, None].cpu().detach()))
        unet_metrics.append(calc_metrics(restored_dict['unet_img'][None, None].cpu().detach(),
                                  IFt(gt_ks).abs()[None, None].cpu().detach()))
        af_unet_metrics.append(calc_metrics(IFt(restored_dict['af_unet_ks']).abs()[None, None].cpu().detach(),
                                  IFt(gt_ks).abs()[None, None].cpu().detach()))
        grad_metric.append(calc_metrics(restored_dict['gradmc_img'].float()[None, None].cpu().detach(),
                                  IFt(gt_ks).abs()[None, None].cpu().detach()))

    stats = {'old_ssim': np.array([d['ssim'] for d in old_metric]),
             'old_psnr': np.array([d['psnr'] for d in old_metric]),
             'old_vif': np.array([d['vif_p'] for d in old_metric]),
             'old_ms_ssim': np.array([d['ms_ssim'] for d in old_metric]),

             'af_ssim': np.array([d['ssim'] for d in af_metrics]),
             'af_psnr': np.array([d['psnr'] for d in af_metrics]),
             'af_vif': np.array([d['vif_p'] for d in af_metrics]),
             'af_ms_ssim': np.array([d['ms_ssim'] for d in af_metrics]),

             'unet_ssim': np.array([d['ssim'] for d in unet_metrics]),
             'unet_psnr': np.array([d['psnr'] for d in unet_metrics]),
             'unet_vif': np.array([d['vif_p'] for d in unet_metrics]),
             'unet_ms_ssim': np.array([d['ms_ssim'] for d in unet_metrics]),

             'grad_ssim': np.array([d['ssim'] for d in grad_metric]),
             'grad_psnr': np.array([d['psnr'] for d in grad_metric]),
             'grad_vif': np.array([d['vif_p'] for d in grad_metric]),
             'grad_ms_ssim': np.array([d['ms_ssim'] for d in grad_metric]),

             'af_unet_ssim': np.array([d['ssim'] for d in af_unet_metrics]),
             'af_unet_psnr': np.array([d['psnr'] for d in af_unet_metrics]),
             'af_unet_vif': np.array([d['vif_p'] for d in af_unet_metrics]),
             'af_unet_ms_ssim': np.array([d['ms_ssim'] for d in af_unet_metrics])}

    df = pd.DataFrame(data=stats)
    df.to_csv(csv_name, mode='a', index=False)