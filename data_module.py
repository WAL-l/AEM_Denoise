#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 16:32
# @Author  : Ws
# @File    : data.py
# @Software: PyCharm
import os
import random

import numpy as np
from torch.utils.data import DataLoader, Dataset
import lightning as L


class NoiseDataset(Dataset):
    def __init__(
            self,
            data_path,
            is_npy=False,
            random_flip=True,
            log=False
    ):
        super().__init__()
        self.is_np = is_npy
        if is_npy:
            self.clean_datas = np.load(os.path.join(data_path, 'data.npy'))
            self.noise_datas = np.load(os.path.join(data_path, 'data_noise.npy'))
        else:
            self.clean_datas = self._load_data(os.path.join(data_path, 'data.txt'))
            self.noise_datas = self._load_data(os.path.join(data_path, 'data_noise.txt'))
        self.random_flip = random_flip
        self.log = log

    def _load_data(self, data_file):
        with open(data_file, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.noise_datas)

    def __getitem__(self, idx):
        if self.is_np:
            noise_data = self.noise_datas[idx]
            clean_data = self.clean_datas[idx]
        else:
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
        noise_data = (noise_data.astype(np.float32))
        clean_data = (clean_data.astype(np.float32))
        return {'noise_data': noise_data, 'clean': clean_data}


class DataModule(L.LightningDataModule):
    def __init__(self, train_dir: str = "", val_dir: str = "", is_npy=True, log=False, batch_size: int = 4,
                 random_flip=True, num_workers: int = 0):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.log = log
        self.batch_size = batch_size
        self.random_flip = random_flip
        self.num_workers = num_workers
        self.is_npy = is_npy

    def setup(self, stage: str):
        self.train_set = NoiseDataset(data_path=self.train_dir, is_npy=self.is_npy, random_flip=self.random_flip, log=self.log)
        self.val_set = NoiseDataset(data_path=self.val_dir, is_npy=self.is_npy, random_flip=self.random_flip, log=self.log)

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

