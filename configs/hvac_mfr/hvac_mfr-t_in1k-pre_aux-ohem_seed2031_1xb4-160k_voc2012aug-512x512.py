_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_1xb4-160k_voc2012aug-512x512.py'

randomness = dict(seed=2031, deterministic=False)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=8000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5))
