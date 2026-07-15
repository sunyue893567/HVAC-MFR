_base_ = './hvac_mfr-stdc2-pre_1xb4-160k_cityscapes-512x1024.py'

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in img_ratios],
            [dict(type='RandomFlip', prob=0., direction='horizontal'),
             dict(type='RandomFlip', prob=1., direction='horizontal')],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')],
        ])
]
