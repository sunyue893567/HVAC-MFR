_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_1xb4-160k_voc2012aug-512x512.py'

# Use the official VOC2012 tree that contains the standard train/aug/val split.
train_data_root = r'D:\.mail\my\Code\s2_corr-main\data\VOCdevkit\VOC2012'
val_data_root = train_data_root
dataset_type = 'PascalVOCDataset'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# Standard VOC trainaug protocol from SegMAN style configs: official train masks
# plus augmented masks, evaluated on VOC2012 val.
train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=train_data_root,
                data_prefix=dict(
                    img_path='JPEGImages',
                    seg_map_path='SegmentationClass'),
                ann_file='ImageSets/Segmentation/train.txt',
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=train_data_root,
                data_prefix=dict(
                    img_path='JPEGImages',
                    seg_map_path='SegmentationClassAug'),
                ann_file='ImageSets/Segmentation/aug.txt',
                pipeline=train_pipeline),
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(data_root=val_data_root))
test_dataloader = val_dataloader

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5))
randomness = dict(seed=3407, deterministic=False)
