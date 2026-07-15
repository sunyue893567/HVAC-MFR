_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_1xb4_voc2012aug-512x512.py'

load_from = r'D:\.mail\my\mmsegmentation\work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_voc2012aug_from-lovasz36k_seed3407\best_mIoU_iter_16000.pth'

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
