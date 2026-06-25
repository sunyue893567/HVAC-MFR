# Copyright (c) OpenMMLab. All rights reserved.
"""HVAC-MFR backbone.

This module adds the Horizontal-Vertical Attention Compression (HVAC)
encoder used by the HVAC-MFR semantic segmentation method. It follows the
MSCAN/SegNeXt stage layout, but replaces the strip-convolution attention with
an explicit horizontal/vertical compressed spatial modulation map.
"""
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

from mmseg.registry import MODELS
from .mscan import Mlp, OverlapPatchEmbed, StemConv


class HVACAttention(BaseModule):
    """Horizontal-Vertical Attention Compression module.

    Given an input feature X, the module compresses X along width and height
    separately, projects the two directional descriptors to one-channel row and
    column responses, forms a 2D modulation map with their outer product, and
    applies residual spatial modulation.
    """

    def __init__(self, channels, axis_kernel_size=7):
        super().__init__()
        assert axis_kernel_size % 2 == 1, 'axis_kernel_size must be odd.'
        axis_padding = axis_kernel_size // 2
        self.horizontal_context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(axis_kernel_size, 1),
                padding=(axis_padding, 0),
                groups=channels,
                bias=True), nn.Conv2d(channels, 1, kernel_size=1))
        self.vertical_context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, axis_kernel_size),
                padding=(0, axis_padding),
                groups=channels,
                bias=True), nn.Conv2d(channels, 1, kernel_size=1))

    def forward(self, x):
        """Forward function."""
        row_context = x.mean(dim=3, keepdim=True)
        col_context = x.mean(dim=2, keepdim=True)
        row_response = torch.sigmoid(self.horizontal_context(row_context))
        col_response = torch.sigmoid(self.vertical_context(col_context))
        modulation = row_response * col_response
        return x + x * modulation


class HVACSpatialAttention(BaseModule):
    """Pointwise projection around HVAC spatial modulation."""

    def __init__(self,
                 in_channels,
                 axis_kernel_size=7,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.hvac = HVACAttention(in_channels, axis_kernel_size)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """Forward function."""
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.hvac(x)
        x = self.proj_2(x)
        return x


class HVACMFRBlock(BaseModule):
    """Encoder block with HVAC and a lightweight convolutional MLP."""

    def __init__(self,
                 channels,
                 axis_kernel_size=7,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = HVACSpatialAttention(channels, axis_kernel_size, act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


@MODELS.register_module()
class HVACMFR(BaseModule):
    """Horizontal-Vertical Attention Compression encoder for HVAC-MFR.

    The stage layout follows SegNeXt/MSCAN so the model can reuse the same MMSeg
    training pipeline while exposing HVAC-MFR as an independent method.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 hvac_axis_kernel_size=7,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(
                    in_channels, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    in_channels=embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg)

            block = nn.ModuleList([
                HVACMFRBlock(
                    channels=embed_dims[i],
                    axis_kernel_size=hvac_axis_kernel_size,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        """Initialize modules of HVAC-MFR."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        """Forward function."""
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs
