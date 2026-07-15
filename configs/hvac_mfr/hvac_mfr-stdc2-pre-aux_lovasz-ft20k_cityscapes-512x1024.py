_base_ = './hvac_mfr-stdc2-pre-aux_1xb4-160k_cityscapes-512x1024.py'

load_from = r'D:\.mail\my\mmsegmentation\work_dirs\hvac_mfr-stdc2-pre-aux_cityscapes_seed3407\best_mIoU_iter_144000.pth'

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
            dict(
                type='LovaszLoss',
                loss_type='multi_class',
                classes='present',
                per_image=False,
                reduction='none',
                loss_weight=0.5),
        ]))

optim_wrapper = dict(optimizer=dict(lr=5e-6))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=250),
    dict(
        type='PolyLR',
        power=1.0,
        begin=250,
        end=20000,
        eta_min=1e-7,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5))
