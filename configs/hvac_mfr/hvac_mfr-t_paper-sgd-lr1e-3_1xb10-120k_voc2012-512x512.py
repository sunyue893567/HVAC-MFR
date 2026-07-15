_base_ = './hvac_mfr-t_1xb4-160k_voc2012-512x512.py'

train_data_root = r'D:\.mail\my\Code\s2_corr-main\data\VOCdevkit\VOC2012'
val_data_root = train_data_root
pretrained_hvac = r'D:\.mail\my\mmsegmentation\pretrain\hvac_mfr_in1k_full.pth'
dataset_type = 'PascalVOCDataset'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.4, 3.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_hvac)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=64,
        num_convs=1,
        num_classes=21,
        in_index=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        concat_input=False,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))

train_dataloader = dict(
    batch_size=10,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=train_data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(data_root=val_data_root))
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0006))

param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        begin=0,
        end=120000,
        eta_min=0.0,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=120000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5))
randomness = dict(seed=3407, deterministic=False)
