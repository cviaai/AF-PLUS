import math
import argparse
import random
import numpy as np
import pandas as pd
import pylab as plt
import torch
import torch.nn.functional as F
import h5py
from torch.utils.tensorboard import SummaryWriter
import piq

from config import PATH

from utils.fastmri import FastMRITransform, DemotionFastMRIh5Dataset
from utils.kspace import RandomMotionTransform
from utils.utils import get_loss_func
from utils.unet import Unet
from utils.kspace import pt_spatial2kspace, pt_kspace2spatial
from tqdm import tqdm
import skimage.data
from utils.utils import t2i, normalize, psnr, ssim, l1_loss

import sys
sys.path.append(PATH.NUFFT_PATH)  # import NUFFT library
import nufft
from torch.fft import fftshift, ifftshift, fftn, ifftn

Ft = lambda x : fftshift(fftn(ifftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
IFt = lambda x : ifftshift(ifftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
from IPython.display import clear_output


random.seed(228)
torch.manual_seed(228)
torch.cuda.manual_seed(228)
np.random.seed(228)

    
def calc_metrics(y_pred: torch.Tensor, y_gt: torch.Tensor):
    metrics_dict = {}
    metrics_dict['psnr'] = psnr(y_pred, y_gt).item()
    metrics_dict['ssim'] = ssim(y_pred, y_gt).item()
    metrics_dict['l1_loss'] = F.l1_loss(y_pred, y_gt).item()
    metrics_dict['ms_ssim'] = piq.multi_scale_ssim(normalize(y_pred),
                                                   normalize(y_gt),
                                                   data_range=1.).item()
    metrics_dict['vif_p'] = piq.vif_p(normalize(y_pred),
                                      normalize(y_gt), data_range=1.).item()
    return metrics_dict


# Algorithm
def get_rot_mat_nufft(rot_vector):
    rot_mat = torch.zeros(rot_vector.shape[0], 2, 2).cuda()
    rot_mat[:, 0, 0] = torch.cos(rot_vector)
    rot_mat[:, 0, 1] = -torch.sin(rot_vector)
    rot_mat[:, 1, 0] = torch.sin(rot_vector)
    rot_mat[:, 1, 1] = torch.cos(rot_vector)
    return rot_mat


def R_differentiable(ks, rot_vector, oversamp=5):
    rot_matrices = get_rot_mat_nufft(rot_vector)
    grid = torch.stack([
        arr.flatten() for arr in torch.meshgrid(
            torch.arange(-ks.shape[0]//2, ks.shape[0]//2).float(),
            torch.arange(-ks.shape[1]//2, ks.shape[1]//2).float(),
            indexing='ij')]).cuda()
    grid = (rot_matrices @ \
            grid.reshape(2, 320, 320).movedim(1, 0)).movedim(0, 1).reshape(2, -1)
    img = nufft.nufft_adjoint(ks, grid.T, device='cuda', oversamp=oversamp, 
                              out_shape=[1, 1, *ks.shape])[0, 0]
    return Ft(img)
    

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nexpr', type=str, help='name of the experiment')
    parser.add_argument('motion_type', type=str, help='motion type that is used')
    parser.add_argument('--t', type=int, default=50, help='train dataset size')
    parser.add_argument('--v', type=int, default=20, help='val dataset size')
    parser.add_argument('--e', type=int, default=100, help='num of epoches')
    parser.add_argument('--verb', type=int, default=5, help='validate every N epoch')
    parser.add_argument('--accum', type=int, default=16, help='gradient accumulation')
    parser.add_argument('--init', type=str, default='none', 
                        help='type of U-Net initialization')
    parser.add_argument('--t_lr', type=float, default=3e-4, 
                        help='learning rate for translation parameters')
    parser.add_argument('--r_lr', type=float, default=3e-4, 
                        help='learning rate for rotation parameters')
    parser.add_argument('--nn_lr', type=float, default=5e-5, 
                        help='learning rate for U-Net')
    parser.add_argument('--train_steps', type=int, default=30, 
                        help='number of steps in training')
    parser.add_argument('--val_steps', type=int, default=80, 
                        help='number of steps in validation')
    parser.add_argument('--loss', type=str, default='ssim', 
                        help='Loss function used for U-Net train')
    args = parser.parse_args()
    return args


def simult_de_motion(ks, gt_ks, args):
    beta1, beta2 = 0.89, 0.8999
    ps = ks.shape[-1]
    ps_cf = int((ps // 2) * 0.08)
    zero_middle = torch.ones((ps)).cuda()
    zero_middle[ps // 2 - ps_cf : ps // 2 + ps_cf] = 0.
    img, gt_img = IFt(ks).abs(), IFt(gt_ks).abs()

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
    
    for j in range(args.val_steps):
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
        x_shifts = x_shifts - args.t_lr * x_moment1 * x_moment2.rsqrt()
        y_shifts = y_shifts - args.t_lr * y_moment1 * y_moment2.rsqrt()
        rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
        rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
        rot_vector = rot_vector - args.r_lr  * rot_moment1 * rot_moment2.rsqrt()
 
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
    old_ks_metrics = calc_metrics(IFt(ks).abs()[None, None].cpu().detach(),
                                  IFt(gt_ks).abs()[None, None].cpu().detach())
    mid_ks_metrics = calc_metrics(IFt(yp_ks).abs()[None, None].cpu().detach(),
                                  IFt(gt_ks).abs()[None, None].cpu().detach())
    new_ks_metrics = calc_metrics(IFt(new_k_space).abs()[None, None].cpu().detach(),
                                  IFt(gt_ks).abs()[None, None].cpu().detach())
    return old_ks_metrics, mid_ks_metrics, new_ks_metrics


def check_simple_algorithm_version(val_dataset, args, verbose=True):
    old_metrics = []
    new_metrics = []

    for batch in tqdm(val_dataset):
        gt_ks = batch['target_k_space']
        gt_ks = gt_ks[0] + 1j * gt_ks[1]
        ks = batch['k_space']
        ks = ks[0] + 1j * ks[1]

        old, _, new = simult_de_motion(ks.cuda(), gt_ks.cuda(), args=args)
        old_metrics.append(old)
        new_metrics.append(new)
        
    auto_ssim_vals = np.array([d['ssim'] for d in new_metrics])
    auto_psnr_vals = np.array([d['psnr'] for d in new_metrics])
    auto_vif_vals = np.array([d['vif_p'] for d in new_metrics])
    auto_ms_ssim_vals = np.array([d['ms_ssim'] for d in new_metrics])
    auto_l1_loss_vals = np.array([d['l1_loss'] for d in new_metrics])
    auto_stats = {'ssim_mean': auto_ssim_vals.mean(),
                  'psnr_mean': auto_psnr_vals.mean(),
                  'vif_mean': auto_vif_vals.mean(),
                  'ms_ssim_mean': auto_ms_ssim_vals.mean(),
                  'l1_loss_mean': auto_l1_loss_vals.mean(),
                  
                  'ssim_std': auto_ssim_vals.std(),
                  'psnr_std': auto_psnr_vals.std(),
                  'vif_std': auto_vif_vals.std(),
                  'ms_ssim_std': auto_ms_ssim_vals.std(),
                  'l1_loss_std': auto_l1_loss_vals.std()}
    
    old_ssim_vals = np.array([d['ssim'] for d in old_metrics])
    old_psnr_vals = np.array([d['psnr'] for d in old_metrics])
    old_vif_vals = np.array([d['vif_p'] for d in old_metrics])
    old_ms_ssim_vals = np.array([d['ms_ssim'] for d in old_metrics])
    old_l1_loss_vals = np.array([d['l1_loss'] for d in old_metrics])
    old_stats = {'ssim_mean': old_ssim_vals.mean(),
                  'psnr_mean': old_psnr_vals.mean(),
                  'vif_mean': old_vif_vals.mean(),
                  'ms_ssim_mean': old_ms_ssim_vals.mean(),
                  'l1_loss_mean': old_l1_loss_vals.mean(),
                  
                  'ssim_std': old_ssim_vals.std(),
                  'psnr_std': old_psnr_vals.std(),
                  'vif_std': old_vif_vals.std(),
                  'ms_ssim_std': old_ms_ssim_vals.std(),
                  'l1_loss_std': old_l1_loss_vals.std()}
    if verbose:
        print('SSIM:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
            old_stats['ssim_mean'], old_stats['ssim_std'], auto_stats['ssim_mean'], auto_stats['ssim_std']))
        print('PSNR:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
            old_stats['psnr_mean'], old_stats['psnr_std'], auto_stats['psnr_mean'], auto_stats['psnr_std']))
        print('VIF:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
            old_stats['vif_mean'], old_stats['vif_std'], auto_stats['vif_mean'], auto_stats['vif_std']))
        print('MS-SSIM:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
            old_stats['ms_ssim_mean'], old_stats['ms_ssim_std'], auto_stats['ms_ssim_mean'], auto_stats['ms_ssim_std']))
        print('L1-Loss:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
            old_stats['l1_loss_mean'], old_stats['l1_loss_std'], auto_stats['l1_loss_mean'], auto_stats['l1_loss_std']))
    return auto_stats, old_stats

    
def load_val_dataset(motion_type, n_item):
        
#     val_data_path = PATH.VAL_PATH + '{}.h5'.format(motion_type)
    val_data_path = PATH.VAL_PATH + PATH.VAL_NAME
    
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
    print('-'*30)
    print('Experiment name:', args.nexpr)
    print('Train dataset = {}; Val dataset = {}; Gradient accumulations = {}'.format(args.t, args.v, args.accum))
    print('Num of train steps = {}, Num of val steps = {}'.format(args.train_steps, args.val_steps))
    print('Translation LR = {}; Rotation LR = {}; NN LR = {}'.format(args.t_lr, args.r_lr, args.nn_lr))
    print('Motion Type:', args.motion_type)
    print('Loss func:', args.loss)
    print('-'*30)
    
    train_dataset = DemotionFastMRIh5Dataset(
        PATH.TRAIN_PATH + PATH.TRAIN_NAME,
        None,
        RandomMotionTransform(xy_max=5, theta_max=1.5, num_motions=5,
                              center_fractions=0.08, wave_num=6,
                              motion_type=args.motion_type, noise_lvl=0),
        z_slices=0.1)
    train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(len(train_dataset))[:args.t])
    val_dataset = load_val_dataset(args.motion_type, args.v)
    
    # Calculate Metrics of Corrupted Dataset
    auto_stats, old_stats = check_simple_algorithm_version(val_dataset, args)
    
    # Run Algorithm with U-Net 
    unet = Unet(1, 1, 32, 6, batchnorm=torch.nn.InstanceNorm2d, init_type=args.init).cuda()
    loss_func = get_loss_func(args.loss)
    optimizer_unet = torch.optim.Adam(unet.parameters(), lr=args.nn_lr, betas=(0.9, 0.999)) 

    beta1, beta2 = 0.89, 0.8999
    
    writer = SummaryWriter(log_dir=PATH.LOG_PATH + args.nexpr)
    # Create blank file for metrics 
    csv_name = PATH.SAVING_PATH + args.nexpr + '.csv'
    with open(csv_name, "w") as f:
        pass
    metric_buf = {'psnr': 20.0,
                  'ssim': 0.4}
    
    for epoch in range(args.e):
        print('-'*20, 'For Epoch: ', epoch, '-'*20)
        # Training
        losses_train = []
        unet.train()
        
        # Shuffle idxs of Data Samples
        shuff_idx = np.arange(args.t)
        np.random.shuffle(shuff_idx)
        
        pbar = tqdm(enumerate(shuff_idx), total=args.t)
        for i, batch_idx in pbar:
            batch = train_dataset[batch_idx]
            gt_ks = batch['target_k_space']
            gt_ks = gt_ks[0] + 1j * gt_ks[1]
            ks = batch['k_space']
            ks = ks[0] + 1j * ks[1]

            ps = ks.shape[-1]
            ps_cf = int(ps//2 * 0.08)
            zero_middle = torch.ones((ps)).cuda()
            zero_middle[ps//2 - ps_cf:ps//2 + ps_cf] = 0.
            img = IFt(ks).abs()
            gt_img = IFt(gt_ks).abs()
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
            for _ in range(args.train_steps):
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
        
                loss_net = (yp_img[None, None] * 1e4 * \
                            unet(yp_img[None, None] * 1e4).sigmoid()).mean()

                x_grad, y_grad, rot_grad = torch.autograd.grad(loss_net, [x_shifts, y_shifts, rot_vector],
                                                               create_graph=True)
                x_grad, y_grad, rot_grad = x_grad * 1e-4, y_grad * 1e-4, rot_grad * 1e-4
                x_moment1 = beta1 * x_moment1.detach() + (1. - beta1) * x_grad
                x_moment2 = beta2 * x_moment2.detach() + (1. - beta2) * x_grad * x_grad + 1e-24
                y_moment1 = beta1 * y_moment1.detach() + (1. - beta1) * y_grad
                y_moment2 = beta2 * y_moment2.detach() + (1. - beta2) * y_grad * y_grad + 1e-24
                rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
                rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
                x_shifts = x_shifts - args.t_lr * x_moment1 * x_moment2.rsqrt()
                y_shifts = y_shifts - args.t_lr * y_moment1 * y_moment2.rsqrt()
                rot_vector = rot_vector - args.r_lr  * rot_moment1 * rot_moment2.rsqrt()
                x_shifts.retain_grad()
                y_shifts.retain_grad()
                rot_vector.retain_grad()
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

            loss_img = loss_func(IFt(yp_ks.cuda()).abs() * 1e4,
                                 IFt(gt_ks.cuda()).abs() * 1e4)
            losses_train.append(loss_img.cpu().item())
            loss_img.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            if i % args.accum == 0 and i != 0:
                optimizer_unet.step()
                optimizer_unet.zero_grad()
            pbar.set_description('loss: {:.4}'.format(loss_img.item()))
        losses_train = np.array(losses_train)
        writer.add_scalar('Train_loss', losses_train.mean(), epoch)
        
        # Validation
        if epoch % args.verb == 0: 
            unet.eval()
            new_metrics = []
            losses_val = []
            idx = 0
            for batch in tqdm(val_dataset):
                gt_ks = batch['target_k_space']
                gt_ks = gt_ks[0] + 1j * gt_ks[1]
                ks = batch['k_space']
                ks = ks[0] + 1j * ks[1]

                ps = ks.shape[-1]
                ps_cf = int(ps//2 * 0.08)
                zero_middle = torch.ones((ps)).cuda()
                zero_middle[ps//2 - ps_cf:ps//2 + ps_cf] = 0.
                img, gt_img = IFt(ks).abs(), IFt(gt_ks).abs()

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

                for _ in range(args.val_steps):
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
                                                                   [x_shifts, y_shifts, rot_vector],
                                                                   create_graph=False)
                    x_grad, y_grad = x_grad * 1e-4, y_grad * 1e-4
                    rot_grad = rot_grad * 1e-4
                    x_moment1 = beta1 * x_moment1.detach() + (1. - beta1) * x_grad
                    x_moment2 = beta2 * x_moment2.detach() + (1. - beta2) * x_grad * x_grad + 1e-24
                    y_moment1 = beta1 * y_moment1.detach() + (1. - beta1) * y_grad
                    y_moment2 = beta2 * y_moment2.detach() + (1. - beta2) * y_grad * y_grad + 1e-24
                    rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
                    rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
                    x_shifts = x_shifts - args.t_lr * x_moment1 * x_moment2.rsqrt()
                    y_shifts = y_shifts - args.t_lr * y_moment1 * y_moment2.rsqrt()
                    rot_vector = rot_vector - args.r_lr * rot_moment1 * rot_moment2.rsqrt()

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

                loss_img = l1_loss(IFt(yp_ks.cuda()).abs() * 1e4,
                                     IFt(gt_ks.cuda()).abs() * 1e4)
                losses_val.append(loss_img.cpu().item())
                new_metrics.append(calc_metrics(IFt(yp_ks).abs().data.cpu()[None, None],
                                                IFt(gt_ks).abs().data.cpu()[None, None]))
                # Log Validation Images
                if epoch % args.verb*2 == 0 and epoch != 0 and idx == 0:
                    img_batch = np.zeros((3, 1, 320, 320))  # normalize [0,1]
                    img_batch[0] = normalize(IFt(ks).abs().cpu().detach()).numpy()[None]
                    img_batch[1] = normalize(IFt(yp_ks).abs().cpu().detach()).numpy()[None]
                    img_batch[2] = normalize(IFt(gt_ks).abs().cpu().detach()).numpy()[None]
                    writer.add_images('validation', img_batch, epoch)
                if epoch % args.verb*2 == 0 and epoch != 0 and idx == 3:
                    img_batch = np.zeros((3, 1, 320, 320))  # normalize [0,1]
                    img_batch[0] = normalize(IFt(ks).abs().cpu().detach()).numpy()[None]
                    img_batch[1] = normalize(IFt(yp_ks).abs().cpu().detach()).numpy()[None]
                    img_batch[2] = normalize(IFt(gt_ks).abs().cpu().detach()).numpy()[None]
                    writer.add_images('validation2', img_batch, epoch)
                if epoch % args.verb*2 == 0 and epoch != 0 and idx == 2:
                    img_batch = np.zeros((3, 1, 320, 320))  # normalize [0,1]
                    img_batch[0] = normalize(IFt(ks).abs().cpu().detach()).numpy()[None]
                    img_batch[1] = normalize(IFt(yp_ks).abs().cpu().detach()).numpy()[None]
                    img_batch[2] = normalize(IFt(gt_ks).abs().cpu().detach()).numpy()[None]
                    writer.add_images('validation3', img_batch, epoch)
                idx += 1

            losses_val = np.array(losses_val)
            ssim_vals = np.array([d['ssim'] for d in new_metrics])
            psnr_vals = np.array([d['psnr'] for d in new_metrics])
            vif_vals = np.array([d['vif_p'] for d in new_metrics])
            ms_ssim_vals = np.array([d['ms_ssim'] for d in new_metrics])
            l1_loss_vals = np.array([d['l1_loss'] for d in new_metrics])

            print('SSIM:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
                old_stats['ssim_mean'], old_stats['ssim_std'],ssim_vals.mean(), ssim_vals.std()))
            print('PSNR:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
                old_stats['psnr_mean'], old_stats['psnr_std'], psnr_vals.mean(), psnr_vals.std()))
            print('VIF:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
                old_stats['vif_mean'], old_stats['vif_std'], vif_vals.mean(), vif_vals.std()))
            print('MS-SSIM:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
                old_stats['ms_ssim_mean'], old_stats['ms_ssim_std'], ms_ssim_vals.mean(), ms_ssim_vals.std()))

            writer.add_scalars('Metric/SSIM', {'corrupted': old_stats['ssim_mean'],
                                               'autofocus': auto_stats['ssim_mean'],
                                              'with_UNet': ssim_vals.mean()}, epoch)

            writer.add_scalars('Metric/PSNR', {'corrupted': old_stats['psnr_mean'],
                                               'autofocus': auto_stats['psnr_mean'],
                                              'with_UNet': psnr_vals.mean()}, epoch)

            writer.add_scalars('Metric/VIF', {'corrupted': old_stats['vif_mean'],
                                               'autofocus': auto_stats['vif_mean'],
                                              'with_UNet': vif_vals.mean()}, epoch)

            writer.add_scalars('Metric/MS-SSIM', {'corrupted': old_stats['ms_ssim_mean'],
                                               'autofocus': auto_stats['ms_ssim_mean'],
                                              'with_UNet': ms_ssim_vals.mean()}, epoch)
            writer.add_scalar('Val_loss', losses_val.mean(), epoch)

            # Save metrics to csv
            df = pd.DataFrame([['{:.5f}'.format(losses_val.mean()),
                                '{:.5f}'.format(old_stats['ssim_mean']) + ' +- ' + '{:.3f}'.format(old_stats['ssim_std']),
                                '{:.5f}'.format(ssim_vals.mean()) + ' +- ' + '{:.3f}'.format(ssim_vals.std()),
                                '{:.5f}'.format(old_stats['psnr_mean']) + ' +- ' + '{:.3f}'.format(old_stats['psnr_std']),
                                '{:.5f}'.format(psnr_vals.mean()) + ' +- ' + '{:.3f}'.format(psnr_vals.std()),
                                '{:.5f}'.format(old_stats['vif_mean']) + ' +- ' + '{:.3f}'.format(old_stats['vif_std']),
                                '{:.5f}'.format(vif_vals.mean()) + ' +- ' + '{:.3f}'.format(vif_vals.std()),
                                '{:.5f}'.format(old_stats['ms_ssim_mean']) + ' +- ' + '{:.3f}'.format(old_stats['ms_ssim_std']),
                                '{:.5f}'.format(ms_ssim_vals.mean()) + ' +- ' + '{:.3f}'.format(ms_ssim_vals.std()),]],
            columns=['loss_val', 'old_ssim_vals +- std', 'ssim_vals +- std', 'old_psnr_vals +- std', 'psnr_vals +- std', 'old_vif + std', 'vif + std', 'old_ms_ssim + std', 'ms_ssim + std',])
            df.to_csv(csv_name, mode='a', header = False, index=False)
            
            # Save Model Weights
            if ssim_vals.mean() > metric_buf['ssim'] and psnr_vals.mean() > metric_buf['psnr']:
                metric_buf['ssim'] = ssim_vals.mean()
                metric_buf['psnr'] = psnr_vals.mean()
                torch.save(unet.state_dict(), PATH.SAVING_PATH + '{}_best.pt'.format(args.nexpr))

            torch.save(unet.state_dict(), PATH.SAVING_PATH + '{}_last.pt'.format(args.nexpr))
    writer.close()
    