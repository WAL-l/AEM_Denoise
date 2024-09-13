#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 16:32
# @Author  : Ws
# @File    : main.py
# @Software: PyCharm

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from lightning import Trainer
from lightning.pytorch.profilers import PyTorchProfiler

from data_module import DataModule
from net import Net

profiler = PyTorchProfiler(dirpath=".", filename="perf_logs")


def train():
    data = DataModule(root_dir='data/AP/real_tem_64/log10', batch_size=8, random_flip=False)
    net = Net()
    # net = torch.compile(net)
    # accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
    # TODO 更改Checkpoint保存策略
    checkpoint_callback = ModelCheckpoint(dirpath="./log/v1",
                                          save_top_k=2,
                                          monitor="val_loss",
                                          save_last=True,
                                          every_n_epochs=50,
                                          save_on_train_epoch_end=True,
                                          filename="v1-{epoch:04d}")
    # 如果使用混合精度，则不需要更改渐变，因为渐变未缩放 在应用剪裁功能之前

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        # precision="16-mixed",
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        max_epochs=500,
        default_root_dir="./log/v1",
        check_val_every_n_epoch=1,
        # profiler=profiler
    )

    # tuner = Tuner(trainer)
    # 自动查找学习率
    # tuner.lr_find(net, datamodule=data)

    trainer.fit(model=net, datamodule=data)


if __name__ == '__main__':
    train()
