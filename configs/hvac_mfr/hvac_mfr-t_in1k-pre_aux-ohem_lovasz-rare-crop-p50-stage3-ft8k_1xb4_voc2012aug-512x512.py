_base_ = './hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_1xb4_voc2012aug-512x512.py'

custom_imports = dict(
    imports=['rare_class_crop'], allow_failed_imports=False)

load_from = r'D:\.mail\my\mmsegmentation\work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-stage2-ft20k_voc2012aug_from-lovasz36k_seed3407\best_mIoU_iter_16000.pth'

train_data_root = r'D:\.mail\my\2012\VOCdevkit\VOC2012'
dataset_type = 'PascalVOCDataset'
crop_size = (512, 512)
target_classes = [2, 4, 9, 11, 16, 18, 20]

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
        crop_size=crop_size,
        target_classes=target_classes,
        target_prob=0.5,
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

optim_wrapper = dict(optimizer=dict(lr=1e-6))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=100),
    dict(
        type='PolyLR',
        power=1.0,
        begin=100,
        end=8000,
        eta_min=1e-8,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=5))
randomness = dict(seed=3407, deterministic=False)
