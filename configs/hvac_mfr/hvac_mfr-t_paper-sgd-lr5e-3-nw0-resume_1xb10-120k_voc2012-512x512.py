_base_ = './hvac_mfr-t_paper-sgd-lr5e-3_1xb10-120k_voc2012-512x512.py'

# Windows pagefile is too small on the remote host. Avoid spawning dataloader
# workers that each import torch/cv2 and trigger WinError 1455.
train_dataloader = dict(num_workers=0, persistent_workers=False)
val_dataloader = dict(num_workers=0, persistent_workers=False)
test_dataloader = val_dataloader

