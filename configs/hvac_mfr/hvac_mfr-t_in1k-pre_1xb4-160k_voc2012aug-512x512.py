_base_ = ['./hvac_mfr-t_1xb4-160k_voc2012-512x512.py']

# PASCAL VOC train+aug protocol:
# - train split uses official masks in SegmentationClass
# - aug split uses SBD/VOC augmented masks in SegmentationClassAug
# - validation remains the official VOC2012 val split
train_data_root = r'D:\.mail\my\2012\VOCdevkit\VOC2012'
val_data_root = r'D:\.mail\my\Code\s2_corr-main\data\VOCdevkit\VOC2012'
pretrained_hvac = r'D:\.mail\my\mmsegmentation\pretrain\hvac_mfr_in1k_full.pth'
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

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_hvac)))

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
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
    num_workers=4,
    dataset=dict(data_root=val_data_root))
test_dataloader = val_dataloader
