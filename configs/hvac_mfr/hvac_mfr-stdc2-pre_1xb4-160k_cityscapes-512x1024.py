_base_ = './hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py'

# ImageNet-pretrained STDC2 backbone (required to reach ~79 mIoU on Cityscapes).
checkpoint = 'pretrain/stdc2_20220308-7dbd9127.pth'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))))
