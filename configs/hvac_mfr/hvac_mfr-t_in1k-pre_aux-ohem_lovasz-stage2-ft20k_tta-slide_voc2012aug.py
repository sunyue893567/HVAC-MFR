_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_1xb4_voc2012aug-512x512.py'

model = dict(test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')],
        ])
]
