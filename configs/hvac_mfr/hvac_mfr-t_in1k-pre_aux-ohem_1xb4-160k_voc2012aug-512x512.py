_base_ = './hvac_mfr-t_in1k-pre_1xb4-160k_voc2012aug-512x512.py'

norm_cfg = dict(type='BN', requires_grad=True)
ohem_sampler = dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)

model = dict(
    decode_head=dict(sampler=ohem_sampler),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=160,
            channels=64,
            num_convs=1,
            num_classes=21,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=ohem_sampler,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=21,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=ohem_sampler,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ])
