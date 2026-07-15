_base_ = './hvac_mfr-stdc2-pre_1xb4-160k_cityscapes-512x1024.py'

# Add back the standard STDC auxiliary supervision (deep-supervision FCN heads
# + STDCHead detail/boundary boosting loss) that the canonical STDC recipe uses
# to reach ~79 mIoU on Cityscapes.
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128, channels=64, num_convs=1, num_classes=19,
            in_index=2, norm_cfg=norm_cfg, concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128, channels=64, num_convs=1, num_classes=19,
            in_index=1, norm_cfg=norm_cfg, concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='STDCHead',
            in_channels=256, channels=64, num_convs=1, num_classes=2,
            boundary_threshold=0.1, in_index=0, norm_cfg=norm_cfg,
            concat_input=False, align_corners=True,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=True, loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
            ]),
    ])
