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
from k_space_reconstruction.utils.metrics import psnr, ssim
import piq

from k_space_reconstruction.datasets.fastmri import FastMRITransform, DemotionFastMRIh5Dataset, RandomMaskFunc
from k_space_reconstruction.utils.kspace import RandomMotionTransform
from k_space_reconstruction.nets.unet import Unet
from k_space_reconstruction.utils.kspace import pt_spatial2kspace, pt_kspace2spatial
from tqdm import tqdm
import skimage.data
from k_space_reconstruction.utils.metrics import l1_loss

import sys
sys.path.append('./pytorch_nufft')
import nufft
from torch.fft import fftshift, ifftshift, fftn, ifftn

Ft = lambda x : fftshift(fftn(ifftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
IFt = lambda x : ifftshift(ifftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

random.seed(228)
torch.manual_seed(228)
torch.cuda.manual_seed(228)
np.random.seed(228)


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


# Algorithm
def get_rot_mat_nufft(rot_vector):
    rot_mat = torch.zeros(rot_vector.shape[0], 2, 2).cuda()
    rot_mat[:, 0, 0] = torch.cos(rot_vector)
    rot_mat[:, 0, 1] = -torch.sin(rot_vector)
    rot_mat[:, 1, 0] = torch.sin(rot_vector)
    rot_mat[:, 1, 1] = torch.cos(rot_vector)
    return rot_mat


def R_differentiable(ks, rot_vector):
    rot_matrices = get_rot_mat_nufft(rot_vector)
    grid = torch.stack([
        arr.flatten() for arr in torch.meshgrid(
            torch.arange(-ks.shape[0]//2, ks.shape[0]//2).float(),
            torch.arange(-ks.shape[1]//2, ks.shape[1]//2).float(),
            indexing='ij')]).cuda()
    new_grid = grid.clone()
    for i in range(ks.shape[0]):
        new_grid[:, ks.shape[1] * i : ks.shape[1] * (i+1)] = \
        rot_matrices[i] @ grid[:, ks.shape[1] * i : ks.shape[1] * (i+1)]
    
    img = nufft.nufft_adjoint(ks, new_grid.T, device='cuda', oversamp=5, out_shape=[1, 1, *ks.shape])[0, 0]
    return Ft(img)


def l1_loss(pred_y, gt_y):
    return ((t2i(gt_y) - t2i(pred_y)).abs()).sum() / torch.numel(pred_y)

def compund_ssim_l1_loss(pred, gt):
    from pytorch_msssim import ssim
    f1 = l1_loss(pred, gt)
    return (1 - 0.84) * f1 + 0.84 * (1 - ssim(gt[None,None], pred[None,None],
                                              size_average=True,
                                              nonnegative_ssim=True))

def ssim_loss(pred_y, gt_y):
    return 1 - ssim(pred_y, gt_y)
    
def get_loss_func(loss_name):
    if loss_name == 'l1':
        loss = l1_loss
    elif loss_name == 'ssim':
        loss = ssim_loss
    elif loss_name == 'ssim_l1':
        loss = compund_ssim_l1_loss
    else:
        raise ValueError('Incorrect loss function name :(') 
        
    return loss
    

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nexpr', type=str, help='name of the experiment')
    parser.add_argument('motion_type', type=str, help='motion type that is used')
    parser.add_argument('--t', type=int, default=50, help='train dataset size')
    parser.add_argument('--v', type=int, default=20, help='val dataset size')
    parser.add_argument('--e', type=int, default=100, help='num of epoches')
    parser.add_argument('--verb', type=int, default=5, help='validate every N epoch')
    parser.add_argument('--accum', type=int, default=16, help='gradient accumulation')
    parser.add_argument('--init', type=str, default='none', help='type of U-Net initialization')
    
    parser.add_argument('--t_lr', type=float, default=3e-4, help='learning rate for translation parameters')
    parser.add_argument('--r_lr', type=float, default=3e-4, help='learning rate for rotation parameters')
    parser.add_argument('--train_steps', type=int, default=30, help='number of steps in training')
    parser.add_argument('--val_steps', type=int, default=80, help='number of steps in validation')

    parser.add_argument('--loss', type=str, default='ssim', help='Loss function used for U-Net train')
    
    args = parser.parse_args()
    return args


def check_simple_algorithm_version(val_dataset):
    old_metrics = []

    for batch in tqdm(val_dataset):
        gt_ks = batch['target_k_space']
        gt_ks = gt_ks[0] + 1j * gt_ks[1]
        ks = batch['k_space']
        ks = ks[0] + 1j * ks[1]

        old = calc_metrics(IFt(ks).abs()[None, None], IFt(gt_ks).abs()[None, None])
        old_metrics.append(old)

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
    return old_stats


def load_val_dataset(motion_type, n_item):
    
    if motion_type == 'randomize':
        val_data_path = '/home/ekuzmina/fastmri-demotion/datasets/randomized_harmonic_harder.h5'
        shift_vector = torch.zeros((2, 320))
        rot_vector = torch.zeros((1, 320))
    else:
        raise ValueError('Incorrect motion type')
        
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
    print('Motion Type:', args.motion_type)
    print('Loss func:', args.loss)
    print('-'*30)
    
    train_data_path = '/home/a_razumov/small_datasets/small_fastMRIh5_PD_3T/train_small_PD_3T.h5'
    val_data_path = '/home/a_razumov/small_datasets/small_fastMRIh5_PD_3T/val_small_PD_3T.h5'
        
    train_dataset = DemotionFastMRIh5Dataset(
        train_data_path,
        None,
        RandomMotionTransform(xy_max=5, theta_max=1.5, num_motions=5,
                              center_fractions=0.08, wave_num=6,
                              motion_type=args.motion_type, noise_lvl=0),
        z_slices=0.1)
    train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(len(train_dataset))[:args.t])
    
    if args.motion_type == 'randomize':
        val_dataset = load_val_dataset(args.motion_type, args.v)
    else: 
        val_dataset = DemotionFastMRIh5Dataset(
            val_data_path,
            None,
            RandomMotionTransform(xy_max=5, theta_max=1.5, num_motions=5,
                                  center_fractions=0.08, wave_num=6,
                                  motion_type=args.motion_type, noise_lvl=0),
            z_slices=0.1)
        val_dataset = torch.utils.data.Subset(val_dataset, torch.arange(len(val_dataset))[:args.v])
        
    # Calculate Metrics of Corrupted Dataset
    old_stats = check_simple_algorithm_version(val_dataset)
    
    # Run Algorithm with U-Net 
    unet = Unet(1, 1, 32, 6, batchnorm=torch.nn.InstanceNorm2d, init_type=args.init).cuda()
    loss_func = get_loss_func(args.loss)
    optimizer_unet = torch.optim.Adam(unet.parameters(), lr=3e-4, betas=(0.9, 0.999)) 

    beta1, beta2 = 0.89, 0.8999
    writer = SummaryWriter(log_dir='runs/' + args.nexpr)
    
    metric_buf = {'psnr': 20.0,
                  'ssim': 0.4}
    
    for epoch in range(args.e):
        print('-'*20, 'For Epoch: ', epoch, '-'*20)
        # Training
        losses_train = []
        unet.train()
        unet.zero_grad()
        
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
            
            img = IFt(ks).abs().cuda()
            gt_img = IFt(gt_ks).abs().cuda()

            y_pred = unet(img[None, None].cuda())

            loss_img = loss_func(y_pred[0][0], gt_img)
            losses_train.append(loss_img.cpu().item())

            loss_img.backward()
#             torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
#             if i % args.accum == 0 and i != 0:
            optimizer_unet.step()
            optimizer_unet.zero_grad()
        
            pbar.set_description('loss: {:.4}'.format(loss_img.item()))
        losses_train = np.array(losses_train)
        writer.add_scalar('Train_loss', losses_train.mean(), epoch)
        
        # Validation-------------------------------------------------------------------
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

                img = IFt(ks).abs().cuda()
                gt_img = IFt(gt_ks).abs().cuda()
                
                y_pred = unet(img[None, None].cuda())

                loss_img = l1_loss(y_pred[0][0], gt_img)
                losses_val.append(loss_img.cpu().item())
                new_metrics.append(calc_metrics(y_pred.data.cpu(),
                                                gt_img.data.cpu()[None, None]))
                # Log Validation Images
                if epoch % args.verb*2 == 0 and epoch != 0 and idx == 0:
                    img_batch = np.zeros((3, 1, 320, 320))  # normalize [0,1]
                    img_batch[0] = normalize(IFt(ks).abs().cpu().detach()).numpy()[None]
                    img_batch[1] = normalize(y_pred.cpu().detach()).numpy()[None]
                    img_batch[2] = normalize(gt_img.cpu().detach()).numpy()[None]
                    writer.add_images('validation', img_batch, epoch)
                if epoch % args.verb*2 == 0 and epoch != 0 and idx == 3:
                    img_batch = np.zeros((3, 1, 320, 320))  # normalize [0,1]
                    img_batch[0] = normalize(IFt(ks).abs().cpu().detach()).numpy()[None]
                    img_batch[1] = normalize(y_pred.cpu().detach()).numpy()[None]
                    img_batch[2] = normalize(gt_img.cpu().detach()).numpy()[None]
                    writer.add_images('validation2', img_batch, epoch)
                if epoch % args.verb*2 == 0 and epoch != 0 and idx == 2:
                    img_batch = np.zeros((3, 1, 320, 320))  # normalize [0,1]
                    img_batch[0] = normalize(IFt(ks).abs().cpu().detach()).numpy()[None]
                    img_batch[1] = normalize(y_pred.cpu().detach()).numpy()[None]
                    img_batch[2] = normalize(gt_img.cpu().detach()).numpy()[None]
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
    #         print('L1-Loss:\n\tmotion: {:.7f} +- {:.5f}\tL1M: {:.7f} +- {:.5f}'.format(
    #             old_stats['l1_loss_mean'], old_stats['l1_loss_std'], l1_loss_vals.mean(), l1_loss_vals.std()))

            writer.add_scalars('Metric/SSIM', {'corrupted': old_stats['ssim_mean'],
                                              'with_UNet': ssim_vals.mean()}, epoch)

            writer.add_scalars('Metric/PSNR', {'corrupted': old_stats['psnr_mean'],
                                              'with_UNet': psnr_vals.mean()}, epoch)

            writer.add_scalars('Metric/VIF', {'corrupted': old_stats['vif_mean'],
                                              'with_UNet': vif_vals.mean()}, epoch)

            writer.add_scalars('Metric/MS-SSIM', {'corrupted': old_stats['ms_ssim_mean'],
                                              'with_UNet': ms_ssim_vals.mean()}, epoch)

    #         writer.add_scalars('Metric/L1-Loss', {'corrupted': old_stats['l1_loss_mean'],
    #                                            'autofocus': auto_stats['l1_loss_mean'],
    #                                           'with_UNet': l1_loss_vals.mean()}, epoch)
            writer.add_scalar('Val_loss', losses_val.mean(), epoch)

            if ssim_vals.mean() > metric_buf['ssim'] and psnr_vals.mean() > metric_buf['psnr']:
                metric_buf['ssim'] = ssim_vals.mean()
                metric_buf['psnr'] = psnr_vals.mean()
                torch.save(unet, 'experiment_data/{}_best.pt'.format(args.nexpr))
            torch.save(unet, 'experiment_data/{}_last.pt'.format(args.nexpr))
            
    writer.close()