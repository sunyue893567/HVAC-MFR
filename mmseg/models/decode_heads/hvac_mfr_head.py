# Copyright (c) OpenMMLab. All rights reserved.
"""HVAC-MFR decode head.

The head implements Modulated Feature Refinement Block (MFRB), composed of
Global Context-Aware Modulation (GCAM) and Guided Structure-Aware Modulation
(GSAM), then fuses multi-stage encoder features for semantic segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead


class GlobalContextAwareModulation(nn.Module):
    """GCAM with ASPP context extraction and depthwise refinement."""

    def __init__(self,
                 channels,
                 dilations=(1, 6, 12, 18),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                channels,
                channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.aspp_modules = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                self.aspp_modules.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.aspp_modules.append(
                    DepthwiseSeparableConvModule(
                        channels,
                        channels,
                        3,
                        dilation=dilation,
                        padding=dilation,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        self.context_bottleneck = ConvModule(
            (len(dilations) + 1) * channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.depthwise_refine = ConvModule(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        """Forward function."""
        context = [
            resize(
                self.image_pool(x),
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        context.extend([aspp(x) for aspp in self.aspp_modules])
        context = self.context_bottleneck(torch.cat(context, dim=1))
        weights = torch.sigmoid(F.relu(self.depthwise_refine(context)))
        return x * weights


class GuidedStructureAwareModulation(nn.Module):
    """GSAM for low-level boundary and local-structure modulation."""

    def __init__(self,
                 channels,
                 structure_channels=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        structure_channels = structure_channels or max(channels // 2, 16)
        self.pconv1 = ConvModule(
            channels,
            structure_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.local_structure = DepthwiseSeparableConvModule(
            structure_channels,
            structure_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.pconv2 = nn.Conv2d(structure_channels, channels, 1)

    def forward(self, x):
        """Forward function."""
        structure = self.pconv1(x)
        structure = self.local_structure(structure)
        weights = torch.sigmoid(self.pconv2(structure))
        return x * weights


class ModulatedFeatureRefinementBlock(nn.Module):
    """MFRB that fuses high-level semantic and low-level structure features."""

    def __init__(self,
                 high_channels,
                 low_channels,
                 channels,
                 dilations=(1, 6, 12, 18),
                 structure_channels=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.high_align = ConvModule(
            high_channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.low_align = ConvModule(
            low_channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gcam = GlobalContextAwareModulation(
            channels,
            dilations=dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners)
        self.gsam = GuidedStructureAwareModulation(
            channels,
            structure_channels=structure_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, high_feat, low_feat, target_size):
        """Forward function."""
        high_feat = self.high_align(high_feat)
        low_feat = self.low_align(low_feat)
        if high_feat.shape[2:] != target_size:
            high_feat = resize(
                high_feat,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners)
        if low_feat.shape[2:] != target_size:
            low_feat = resize(
                low_feat,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners)
        semantic = self.gcam(high_feat)
        structure = self.gsam(low_feat)
        return semantic + structure


@MODELS.register_module()
class HVACMFRHead(BaseDecodeHead):
    """Segmentation head for Horizontal-Vertical Attention Compression and
    Modulated Feature Refinement.
    """

    def __init__(self,
                 dilations=(1, 6, 12, 18),
                 structure_channels=None,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(self.in_channels) >= 2, \
            'HVACMFRHead expects at least two encoder stages.'
        self.dilations = dilations
        self.structure_channels = structure_channels
        self.low_project = ConvModule(
            self.in_channels[0],
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.refine_blocks = nn.ModuleList([
            ModulatedFeatureRefinementBlock(
                high_channels=high_channels,
                low_channels=self.in_channels[0],
                channels=self.channels,
                dilations=dilations,
                structure_channels=structure_channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                align_corners=self.align_corners)
            for high_channels in self.in_channels[1:]
        ])
        self.fusion = ConvModule(
            self.channels * len(self.in_channels),
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        low_feat = inputs[0]
        target_size = low_feat.shape[2:]
        refined_feats = [self.low_project(low_feat)]
        for idx, block in enumerate(self.refine_blocks, start=1):
            refined_feats.append(block(inputs[idx], low_feat, target_size))
        output = self.fusion(torch.cat(refined_feats, dim=1))
        output = self.cls_seg(output)
        return output
