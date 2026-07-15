_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_1xb4_voc2012aug-512x512.py'

load_from = r'D:\.mail\my\mmsegmentation\work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_voc2012aug_from-lovasz36k_seed3407\best_mIoU_iter_16000.pth'

norm_cfg = dict(type='BN', requires_grad=True)
ohem_sampler = dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
class_weight = [
    0.3003, 1.1105, 1.1683, 1.1244, 1.2968, 1.4627, 0.9527,
    0.7550, 0.5891, 0.9264, 1.4159, 1.0278, 0.6243, 1.1110,
    0.9775, 0.3797, 1.3437, 1.3611, 0.9610, 0.9216, 1.1901
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.5,
                class_weight=class_weight),
            dict(
                type='LovaszLoss',
                loss_type='multi_class',
                reduction='none',
                loss_name='loss_lovasz',
                loss_weight=0.5),
        ]),
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
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4,
                class_weight=class_weight)),
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
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4,
                class_weight=class_weight)),
    ])

optim_wrapper = dict(optimizer=dict(lr=1e-6))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=250),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=250,
        end=8000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=1000)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=1000, save_best='mIoU', max_keep_ckpts=5))
