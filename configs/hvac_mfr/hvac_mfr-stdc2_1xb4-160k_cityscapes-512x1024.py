custom_imports = dict(
    imports=['mmseg.models.decode_heads.hvac_mfr_head'],
    allow_failed_imports=False)

_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
    '../_base_/datasets/cityscapes.py'
]

# HVAC-MFR decode head with STDC2 backbone for Cityscapes.
# This config is used to evaluate the proposed MFRB-style refinement head
# under an STDC2 lightweight backbone setting.
norm_cfg = dict(type='BN', requires_grad=True)
head_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 1024)
data_root = 'data/cityscapes'

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='STDCContextPathNet',
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet2',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=False),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4)),
    decode_head=dict(
        type='HVACMFRHead',
        in_channels=[256, 128, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=128,
        dilations=(1, 6, 12, 18),
        structure_channels=64,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=head_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(data_root=data_root))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'decode_head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False)
]
