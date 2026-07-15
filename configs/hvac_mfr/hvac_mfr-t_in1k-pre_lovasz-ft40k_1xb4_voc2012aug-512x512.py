_base_ = './hvac_mfr-t_in1k-pre_1xb4-160k_voc2012aug-512x512.py'

load_from = r'D:\.mail\my\mmsegmentation\work_dirs\hvac_mfr-t_in1k-pre_voc2012aug_table6_seed42\iter_160000.pth'

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

optim_wrapper = dict(optimizer=dict(lr=1e-5))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=500,
        end=40000,
        eta_min=1e-7,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5))
