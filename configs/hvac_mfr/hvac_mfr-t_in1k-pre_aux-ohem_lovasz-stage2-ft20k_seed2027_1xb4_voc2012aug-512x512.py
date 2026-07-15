_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_1xb4_voc2012aug-512x512.py'

randomness = dict(seed=2027, deterministic=False)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=1000, save_best='mIoU', max_keep_ckpts=5))
