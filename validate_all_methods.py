import math
import argparse
import random
import numpy as np
import pylab as plt
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
import h5py
import piq
import pandas as pd

from k_space_reconstruction.nets.unet import Unet
from k_space_reconstruction.utils.kspace import pt_spatial2kspace, pt_kspace2spatial
from tqdm import tqdm
import skimage.data

from k_space_reconstruction.datasets.fastmri import FastMRITransform, DemotionFastMRIh5Dataset, RandomMaskFunc
from k_space_reconstruction.utils.kspace import RandomMotionTransform
from k_space_reconstruction.nets.unet import Unet

import sys
sys.path.append('./pytorch_nufft')
import nufft
from torch.fft import fftshift, ifftshift, fftn, ifftn

# All functions are validated by me
from full_demotion_unet_param_test import load_val_dataset, check_max_possible_demotion
from full_demotion_unet_param_test import simult_de_motion, R_differentiable

Ft = lambda x : fftshift(fftn(ifftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
IFt = lambda x : ifftshift(ifftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
from IPython.display import clear_output


def t2i(t):
    q = t - t.min()
    w = q / t.max()
    return w * 255

def psnr(img1, img2):
    mse = torch.mean((t2i(img1) - t2i(img2)) ** 2)
    return 20 * torch.log10(255. / torch.sqrt(mse))

def ssim(img1, img2):
    from pytorch_msssim import ssim
    return ssim(t2i(img1)[None,None], t2i(img2)[None,None])

def normalize(x):
    x1 = x - x.min()
    return x1 / x1.max()
    
def calc_metrics(y_pred: torch.Tensor, y_gt: torch.Tensor):
    metrics_dict = {}
    metrics_dict['psnr'] = psnr(y_pred, y_gt).item()
    metrics_dict['ssim'] = ssim(y_pred, y_gt).item()
    metrics_dict['l1_loss'] = F.l1_loss(y_pred, y_gt).item()
    metrics_dict['ms_ssim'] = piq.multi_scale_ssim(normalize(y_pred), normalize(y_gt), data_range=1.).item()
    metrics_dict['vif_p'] = piq.vif_p(normalize(y_pred), normalize(y_gt), data_range=1.).item()
    
    return metrics_dict

def l1_loss(pred_y, gt_y):
    return ((t2i(gt_y) - t2i(pred_y)).abs()).sum() / torch.numel(pred_y)


def just_unet(ks, model):
    unet = torch.load(model).cuda()
    unet.eval()

    img = IFt(ks).abs().cuda()
    y_pred = unet(img[None, None].cuda())
    return y_pred[0][0]


def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('motion_type', type=str, help='motion type that is used')
    
    parser.add_argument('--seed', type=int, default=228, help='random seed')

    args = parser.parse_args()
    return args


def af(ks):
    from full_demotion_unet_param_test import R_differentiable

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
    
    for j in range(80):
        rot_vector = rot_vector * zero_middle
        x_shifts = x_shifts * zero_middle
        y_shifts = y_shifts * zero_middle
        # Translation
        phase_shift = -2 * math.pi * (
            x_shifts * torch.linspace(0, 320, 320)[None, :, None].cuda() + 
            y_shifts * torch.linspace(0, 320, 320)[None, None, :].cuda())[0]
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
        x_shifts = x_shifts - 3e-4 * x_moment1 * x_moment2.rsqrt()  # 2e-3
        y_shifts = y_shifts - 3e-4 * y_moment1 * y_moment2.rsqrt()  # 2e-3
        rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
        rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
        rot_vector = rot_vector - 3e-4  * rot_moment1 * rot_moment2.rsqrt()  # 4e-3
 
    rot_vector = rot_vector * zero_middle
    x_shifts = x_shifts * zero_middle
    y_shifts = y_shifts * zero_middle
    # Translation
    phase_shift = -2 * math.pi * (
        x_shifts * torch.linspace(0, 320, 320)[None, :, None].cuda() + 
        y_shifts * torch.linspace(0, 320, 320)[None, None, :].cuda())[0]
    yp_ks = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
    # Rotation
    new_k_space = R_differentiable(yp_ks, rot_vector)
    
    return new_k_space


def grad_mc(ks):
    from oct2py import octave
    octave.addpath('/home/a_razumov/opt/GradMC2/code/')
    octave.addpath('/home/a_razumov/opt/GradMC2/code/@matFastFFTmotion/private')
    octave.addpath('/home/a_razumov/opt/GradMC2')
    octave.warning('off', 'all')
    
    p = torch.from_numpy(octave.myGradMC(ks.numpy())).abs()
    return p

    
def af_unet(ks, model):

    unet = torch.load(model).cuda()
    unet.eval()
    
    beta1, beta2 = 0.89, 0.8999
    ps = ks.shape[-1]
    ps_cf = int(ps//2 * 0.08)
    zero_middle = torch.ones((ps)).cuda()
    zero_middle[ps//2 - ps_cf:ps//2 + ps_cf] = 0.
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
            x_shifts * torch.linspace(0, 320, 320)[None, :, None].cuda() + 
            y_shifts * torch.linspace(0, 320, 320)[None, None, :].cuda())[0]
        new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
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
        x_shifts * torch.linspace(0, 320, 320)[None, :, None].cuda() + 
        y_shifts * torch.linspace(0, 320, 320)[None, None, :].cuda())[0]
    new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
    # Rotation
    yp_ks = R_differentiable(new_k_space, rot_vector)
    
    return yp_ks
 

def run_image_demotion(ks, gt_ks, model_af, model_nn, save=False):
    
    af_unet_restored = af_unet(ks, model_af)  # ks
    unet_restored = just_unet(ks, model_nn)  # img
    af_restored = af(ks)  # ks
    gradmc_restored = grad_mc(ks)  # img

    return {'af_unet_ks': af_unet_restored,
            'af_ks': af_restored,
            'unet_img': unet_restored,
            'gradmc_img': gradmc_restored}


def load_val_dataset(motion_type, n_item):
        
    val_data_path = '/home/ekuzmina/fastmri-demotion/datasets/{}.h5'.format(motion_type)
    print(motion_type, '-'*45)
    
    shift_vector = torch.zeros((2, 320))
    rot_vector = torch.zeros((1, 320))
    
    hf = h5py.File(val_data_path)
    val_dataset = []
    for f in tqdm(sorted(list(hf.keys())[:n_item])):
        batch = hf[f]
        ks = torch.from_numpy(batch[0])
        ks = torch.stack([ks.real, ks.imag]).to(torch.complex64)
        gt_ks = torch.from_numpy(batch[1])
        gt_ks = torch.stack([gt_ks.real, gt_ks.imag]).to(torch.complex64)

        d = {'k_space': ks,
             'target_k_space': gt_ks,
             'rot_vector': rot_vector,
             'phase_shift': shift_vector}
        val_dataset.append(d)  
    return val_dataset


if __name__ == '__main__': 
    args = parsing_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    val_data_path = '/home/a_razumov/small_datasets/small_fastMRIh5_PD_3T/val_small_PD_3T.h5'
    # motion_type = 'randomize_harmonic'
    model_af = 'experiment_data/unet6_randomize_hard_best.pt'
    model_nn = 'experiment_data/just_unet_randomized_2_best.pt'

    csv_name = 'experiment_data/metrics_random_dataset/val_{}_{}.csv'.format(args.motion_type, args.seed)
    with open(csv_name, "w") as f:
        pass

    # val_dataset = DemotionFastMRIh5Dataset(
    #     val_data_path,
    #     None,
    #     RandomMotionTransform(xy_max=5, theta_max=1.5, num_motions=5,
    #                           center_fractions=0.08, wave_num=6,
    #                           motion_type=motion_type, noise_lvl=0), z_slices=0.1)

    # Load dataset
    val_dataset = load_val_dataset(args.motion_type, 10)
    print(len(val_dataset))

    af_metrics = []
    unet_metrics = []
    af_unet_metrics = []
    old_metric = []
    grad_metric = []
    end_idx = len(val_dataset)
    # for batch_idx in tqdm(range(0, end_idx, 2)):
    for batch_idx in tqdm(range(0, end_idx)):
        batch = val_dataset[batch_idx]
        gt_ks = batch['target_k_space']
        gt_ks = gt_ks[0] + 1j * gt_ks[1]
        ks = batch['k_space']
        ks = ks[0] + 1j * ks[1]

        restored_dict = run_image_demotion(ks, gt_ks, model_af, model_nn)

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


    old_ssim_vals = np.array([d['ssim'] for d in old_metric])
    old_psnr_vals = np.array([d['psnr'] for d in old_metric])
    old_vif_vals = np.array([d['vif_p'] for d in old_metric])
    old_ms_ssim_vals = np.array([d['ms_ssim'] for d in old_metric])

    af_ssim_vals = np.array([d['ssim'] for d in af_metrics])
    af_psnr_vals = np.array([d['psnr'] for d in af_metrics])
    af_vif_vals = np.array([d['vif_p'] for d in af_metrics])
    af_ms_ssim_vals = np.array([d['ms_ssim'] for d in af_metrics])

    unet_ssim_vals = np.array([d['ssim'] for d in unet_metrics])
    unet_psnr_vals = np.array([d['psnr'] for d in unet_metrics])
    unet_vif_vals = np.array([d['vif_p'] for d in unet_metrics])
    unet_ms_ssim_vals = np.array([d['ms_ssim'] for d in unet_metrics])

    af_unet_ssim_vals = np.array([d['ssim'] for d in af_unet_metrics])
    af_unet_psnr_vals = np.array([d['psnr'] for d in af_unet_metrics])
    af_unet_vif_vals = np.array([d['vif_p'] for d in af_unet_metrics])
    af_unet_ms_ssim_vals = np.array([d['ms_ssim'] for d in af_unet_metrics])

    grad_ssim_vals = np.array([d['ssim'] for d in grad_metric])
    grad_psnr_vals = np.array([d['psnr'] for d in grad_metric])
    grad_vif_vals = np.array([d['vif_p'] for d in grad_metric])
    grad_ms_ssim_vals = np.array([d['ms_ssim'] for d in grad_metric])

    stats = {'old_ssim': old_ssim_vals,
             'old_psnr': old_psnr_vals,
             'old_vif': old_vif_vals,
             'old_ms_ssim': old_ms_ssim_vals,

             'af_ssim': af_ssim_vals,
             'af_psnr': af_psnr_vals,
             'af_vif': af_vif_vals,
             'af_ms_ssim': af_ms_ssim_vals,

             'unet_ssim': unet_ssim_vals,
             'unet_psnr': unet_psnr_vals,
             'unet_vif': unet_vif_vals,
             'unet_ms_ssim': unet_ms_ssim_vals,

             'grad_ssim': grad_ssim_vals,
             'grad_psnr': grad_psnr_vals,
             'grad_vif': grad_vif_vals,
             'grad_ms_ssim': grad_ms_ssim_vals,

             'af_unet_ssim': af_unet_ssim_vals,
             'af_unet_psnr': af_unet_psnr_vals,
             'af_unet_vif': af_unet_vif_vals,
             'af_unet_ms_ssim': af_unet_ms_ssim_vals}

    df = pd.DataFrame(data=stats)
    df.to_csv(csv_name, mode='a', index=False)