_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_tta_voc2012aug.py'

img_ratios = [1.0, 1.25, 1.5]
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
