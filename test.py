#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 12:45
# @Author  : Ws
# @File    : tes.py
# @Software: PyCharm
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

import numpy as np
import torch
import matplotlib.pyplot as plt

from net import Net


def pre(noise_data):
    model = Net.load_from_checkpoint('./log/last.ckpt')
    model.eval()
    noise_data = noise_data[np.newaxis, :, np.newaxis].astype(np.float32)
    noise_data = torch.tensor(noise_data).cuda()
    with torch.no_grad():
        pred = model(noise_data)
    return pred


if __name__ == '__main__':
    n = np.load('test/noise.npy')
    pre_data = pre(n)
    pre_noise = pre_data[:, :, 1:2].cpu().numpy()
    clean = n - pre_noise[0, :, 0]
    np.save('test/clean.npy', clean)
