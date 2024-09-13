#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 16:32
# @Author  : Ws
# @File    : data.py
# @Software: PyCharm
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 21:31
# @Author  : Ws
# @File    : DataModul.py
# @Software: PyCharm
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import lightning as L

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class NoiseDataset(Dataset):
    def __init__(
            self,
            data_path,
            random_flip=True,
            log=False
    ):
        super().__init__()
        self.clean_datas = self._load_data(os.path.join(data_path, 'data_train.txt'))
        self.noise_datas = self._load_data(os.path.join(data_path, 'data_train_noisy.txt'))
        self.random_flip = random_flip
        self.log = log

    def _load_data(self, data_file):
        with open(data_file, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.noise_datas)

    def __getitem__(self, idx):
        noise_data = np.fromstring(self.noise_datas[idx], sep=' ')
        clean_data = np.fromstring(self.clean_datas[idx], sep=' ')
        if self.log:
            noise_data = np.log10(noise_data)
            clean_data = np.log10(clean_data)

        if self.random_flip and random.random() < 0.5:
            noise_data = noise_data[::-1]
            clean_data = clean_data[::-1]

        noise_data = noise_data[:, np.newaxis]
        clean_data = clean_data[:, np.newaxis]

        d_max = noise_data.max()
        d_min = noise_data.min()
        noise_data = (noise_data - d_min) / (d_max - d_min)
        clean_data = (clean_data - d_min) / (d_max - d_min)
        # 应用归一化公式
        noise_data = (noise_data.astype(np.float32))
        clean_data = (clean_data.astype(np.float32))
        return {'noise_data': noise_data, 'clean': clean_data}


class DataModule(L.LightningDataModule):
    def __init__(self, root_dir: str = "", batch_size: int = 4,
                 random_flip=True, num_workers: int = 0):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.random_flip = random_flip
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_set = NoiseDataset(data_path=self.root_dir, random_flip=self.random_flip)
        self.val_set = NoiseDataset(data_path=self.root_dir, random_flip=self.random_flip)

    def train_dataloader(self):
        ld_train = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
        return ld_train

    def val_dataloader(self):
        ld_val = DataLoader(
            self.val_set,
            num_workers=0,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return ld_val


def plot_data(data):
    """
    Plot the clean and noisy data at the given index.
    """
    clean_data1, noise_data1 = data['clean'], data['noise_data']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot clean data
    ax1.plot(clean_data1[0,:,0], label='Clean Data')
    ax1.set_title('Clean Data')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)

    # Plot noisy data
    ax2.plot(noise_data1[0,:,0], label='Noisy Data')
    ax2.set_title('Noisy Data')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)

    plt.show()


if __name__ == '__main__':
    data_file = 'data/AP/theory_tem_306/train'  # 假设数据文件名为 data.txt
    data_module = DataModule(data_file, batch_size=1, num_workers=0)
    data_module.setup('fit')

    # 创建数据集实例
    dataset = NoiseDataset(data_file)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # 绘制前几个数据的图像
    for idx in train_loader:
        plot_data(idx)
