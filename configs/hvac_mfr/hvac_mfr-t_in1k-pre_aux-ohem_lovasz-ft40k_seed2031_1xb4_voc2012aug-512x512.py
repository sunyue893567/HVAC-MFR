_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_1xb4_voc2012aug-512x512.py'

load_from = r'D:\.mail\my\mmsegmentation\work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_voc2012aug_seed2031\best_mIoU_iter_144000.pth'

randomness = dict(seed=2031, deterministic=False)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5))
