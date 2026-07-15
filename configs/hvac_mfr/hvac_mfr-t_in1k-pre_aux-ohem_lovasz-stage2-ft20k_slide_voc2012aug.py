_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_1xb4_voc2012aug-512x512.py'

model = dict(test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
