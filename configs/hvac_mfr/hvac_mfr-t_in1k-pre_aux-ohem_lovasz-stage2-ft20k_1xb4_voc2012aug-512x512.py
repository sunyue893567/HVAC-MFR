_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_1xb4_voc2012aug-512x512.py'

load_from = r'D:\.mail\my\mmsegmentation\work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_voc2012aug_from-aux160k_seed3407\best_mIoU_iter_36000.pth'

optim_wrapper = dict(optimizer=dict(lr=2e-6))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=250),
    dict(
        type='PolyLR',
        power=1.0,
        begin=250,
        end=20000,
        eta_min=1e-8,
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
