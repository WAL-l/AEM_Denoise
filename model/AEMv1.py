#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/12 19:06
# @Author  : Ws
# @File    : AEMv1.py
# @Software: PyCharm

import math
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import PatchEmbed
from torch.utils.cpp_extension import load

T_MAX = 8192
wkv_cuda = load(name="wkv", sources=["./model/cuda/wkv_op.cpp", "./model/cuda/wkv_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--use_fast_math', '-O3',
                                   f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


# class DensePatchEmbed(nn.Module):
#
#     def __init__(
#             self,
#             signal_size: int = 224,
#             patch_size: int = 16,
#             in_chans: int = 3,
#             embed_dim: int = 768,
#             bias: bool = True,
#     ):
#         super().__init__()
#         self.patch_size = patch_size
#         self.kernel_size = 2*patch_size-1
#         if signal_size is not None:
#             self.signal_size = signal_size
#             self.grid_size = tuple([s // p for s, p in zip(self.signal_size, self.patch_size)])
#             self.num_patches = self.grid_size[0] * self.grid_size[1]
#         else:
#             self.signal_size = None
#             self.grid_size = None
#             self.num_patches = None
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.kernel_size, stride=patch_size, bias=bias)
#         self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         pad_h = (self.kernel_size - H % self.kernel_size) % self.kernel_size
#         pad_w = (self.kernel_size - W % self.kernel_size) % self.kernel_size
#         x = F.pad(x, (0, pad_w, 0, pad_h))
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
#         return self.norm(x)


def q_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    assert gamma <= 1 / 4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :,
                                                                         shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3),
                                                                         0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:,
                                                                             int(C * gamma * 3):int(C * gamma * 4),
                                                                             shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output.flatten(2).transpose(1, 2)




class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, init_mode,
                                    key_norm=key_norm)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, hidden_rate,
                                    init_mode, key_norm=key_norm)
        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x, patch_resolution):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.gamma1 * self.att(self.ln1(x), patch_resolution)
        x = x + self.gamma2 * self.ffn(self.ln2(x), patch_resolution)
        return x


class VRWKV(nn.Module):
    def __init__(self,
                 in_channels=1,
                 embed_dims=256,
                 depth=48,
                 channel_gamma=1 / 4,
                 shift_pixel=1,
                 shift_mode='q_shift',
                 init_mode='fancy',
                 key_norm=False,
                 hidden_rate=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = depth

        self.patch_embed = nn.Linear(in_channels, embed_dims)
        self.patch_resolution = (8, 8)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                init_mode=init_mode,
                key_norm=key_norm,
            ))
        self.fn1 = nn.Linear(embed_dims, embed_dims//2)
        self.fn2 = nn.Linear(embed_dims//2, 2)

    def forward(self, x):
        x = self.patch_embed(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, self.patch_resolution)
        return self.fn2(self.fn1(x))


if __name__ == "__main__":

    x = torch.zeros((1, 64, 1)).type(torch.FloatTensor).cuda()
    model = VRWKV()
    model.cuda()
    y = model(x)
    print(y)
