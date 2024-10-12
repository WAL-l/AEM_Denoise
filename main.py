#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 16:32
# @Author  : Ws
# @File    : main.py
# @Software: PyCharm

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.pytorch.profilers import PyTorchProfiler

from data_module import DataModule
from net import Net

profiler = PyTorchProfiler(dirpath=".", filename="perf_logs")


def train():
    data = DataModule(train_dir='', val_dir='', is_npy=False, batch_size=32,
                      log=False, random_flip=False)
    net = Net()
    # TODO 更改Checkpoint保存策略
    checkpoint_callback = ModelCheckpoint(dirpath="./log",
                                          save_top_k=2,
                                          monitor="val_loss",
                                          save_last=True,
                                          every_n_epochs=50,
                                          save_on_train_epoch_end=True,
                                          filename="v2-{epoch:04d}")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        max_epochs=200,
        default_root_dir="./log",
        check_val_every_n_epoch=1,
    )

    trainer.fit(model=net, datamodule=data)


if __name__ == '__main__':
    train()
