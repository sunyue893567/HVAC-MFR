_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-rare-crop-p50-stage3-ft8k_1xb4_voc2012aug-512x512.py'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RareClassRandomCrop',
        crop_size=(512, 512),
        target_classes=[2, 4, 9, 11, 16, 18, 20],
        target_prob=0.75,
        min_target_pixels=2048,
        max_tries=30,
        cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='PascalVOCDataset',
                data_root=r'D:\.mail\my\2012\VOCdevkit\VOC2012',
                data_prefix=dict(
                    img_path='JPEGImages',
                    seg_map_path='SegmentationClass'),
                ann_file='ImageSets/Segmentation/train.txt',
                pipeline=train_pipeline),
            dict(
                type='PascalVOCDataset',
                data_root=r'D:\.mail\my\2012\VOCdevkit\VOC2012',
                data_prefix=dict(
                    img_path='JPEGImages',
                    seg_map_path='SegmentationClassAug'),
                ann_file='ImageSets/Segmentation/aug.txt',
                pipeline=train_pipeline),
        ]))
