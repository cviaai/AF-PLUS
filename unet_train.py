import math
import argparse
import random
import numpy as np
import pandas as pd
import pylab as plt
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_loss_func
import piq
from utils.fastmri import FastMRITransform, DemotionFastMRIh5Dataset

from utils.unet import Unet
import skimage.data
from utils.metrics import l1_loss

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
    return ssim(t2i(img1)[None, None], t2i(img2)[None, None])


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
    parser.add_argument('--init', type=str, default='none', help='type of U-Net initialization')
    

    parser.add_argument('--train_steps', type=int, default=30, help='number of steps in training')
    parser.add_argument('--val_steps', type=int, default=80, help='number of steps in validation')

    parser.add_argument('--loss', type=str, default='ssim', help='Loss function used for U-Net train')
    
    args = parser.parse_args()
    return args


def load_val_dataset(motion_type, n_item):
        
    val_data_path = '/home/ekuzmina/fastmri-demotion/datasets/{}.h5'.format(motion_type)
    
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
    val_dataset = load_val_dataset(args.motion_type, args.v)
        
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

            writer.add_scalars('Metric/SSIM', {'corrupted': old_stats['ssim_mean'],
                                              'with_UNet': ssim_vals.mean()}, epoch)

            writer.add_scalars('Metric/PSNR', {'corrupted': old_stats['psnr_mean'],
                                              'with_UNet': psnr_vals.mean()}, epoch)

            writer.add_scalars('Metric/VIF', {'corrupted': old_stats['vif_mean'],
                                              'with_UNet': vif_vals.mean()}, epoch)

            writer.add_scalars('Metric/MS-SSIM', {'corrupted': old_stats['ms_ssim_mean'],
                                              'with_UNet': ms_ssim_vals.mean()}, epoch)

            writer.add_scalar('Val_loss', losses_val.mean(), epoch)

            if ssim_vals.mean() > metric_buf['ssim'] and psnr_vals.mean() > metric_buf['psnr']:
                metric_buf['ssim'] = ssim_vals.mean()
                metric_buf['psnr'] = psnr_vals.mean()
                torch.save(unet, 'experiment_data/{}_best.pt'.format(args.nexpr))
            torch.save(unet, 'experiment_data/{}_last.pt'.format(args.nexpr))
            
    writer.close()