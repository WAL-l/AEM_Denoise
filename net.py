#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 16:31
# @Author  : Ws
# @File    : anet.py
# @Software: PyCharm
import lightning as L
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from torch import nn

from model.AEMv1 import VRWKV


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class Net(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.net = VRWKV()
        self.compute_loss = nn.MSELoss()
        # self.w_loss = AutomaticWeightedLoss(2)
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):

        out = self.forward(batch['noise_data'])
        clean_loss = self.compute_loss(out[:, :, 0:1], batch['clean'])

        noise = batch['noise_data'] - batch['clean']
        noise_loss = self.compute_loss(out[:, :, 1:2], noise)

        loss = clean_loss + noise_loss
        self.log_dict({'train_loss': loss, 'clean_loss': clean_loss, 'noise_loss': noise_loss},
                      prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch['noise_data'])
        clean_loss = self.compute_loss(out[:, :, 0:1], batch['clean'])

        noise = batch['noise_data'] - batch['clean']
        noise_loss = self.compute_loss(out[:, :, 1:2], noise)

        loss = clean_loss + noise_loss

        noise_clean = batch['noise_data'] - out[:, :, 1:2]

        self.save_val_fig(out[0, :, 0].cpu().numpy(),noise_clean[0, :, 0].cpu().numpy(), './log/v1')
        self.log_dict(
            {'val_loss': loss, 'psnr': psnr(out[0, :, 0].cpu().numpy(), batch['clean'][0, :, 0].cpu().numpy(), data_range=2),
                               'ssim': ssim(out[0, :, 0].cpu().numpy(), batch['clean'][0, :, 0].cpu().numpy(), data_range=2)}, prog_bar=True)
        self.log_dict(
            {'psnr_noise': psnr(noise_clean[0, :, 0].cpu().numpy(), batch['clean'][0, :, 0].cpu().numpy(), data_range=2),
             'ssim_noise': ssim(noise_clean[0, :, 0].cpu().numpy(), batch['clean'][0, :, 0].cpu().numpy(), data_range=2)},
            prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))

    def save_val_fig(self, data1, data2, path):
        clean_data1, noise_data1 = data1, data2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot clean data
        ax1.plot(clean_data1, label='Clean Data')
        ax1.set_title('Clean Data')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)

        # Plot noisy data
        ax2.plot(noise_data1, label='Noisy Data')
        ax2.set_title('Noisy-clean Data')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)

        plt.savefig(path)

