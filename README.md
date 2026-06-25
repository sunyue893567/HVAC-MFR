# HVAC-MFR

Official implementation of **HVAC-MFR**, a lightweight semantic segmentation framework with horizontal-vertical attention compression and modulated feature refinement.

## Repository contents

- `mmseg/models/backbones/hvac_mfr.py`: HVAC-MFR lightweight encoder.
- `mmseg/models/decode_heads/hvac_mfr_head.py`: HVAC-MFR decode head with GCAM/GSAM refinement.
- `configs/hvac_mfr/`: training configurations for Cityscapes and PASCAL VOC 2012.
- `docs/registration.md`: optional registration notes for MMSegmentation.

## Environment

The experiments were conducted with the following software environment:

| Component | Version |
|---|---|
| Python | 3.10.8 |
| PyTorch | 2.1.2+cu118 |
| TorchVision | 0.16.2+cu118 |
| CUDA runtime | 11.8 |
| MMCV | 2.2.0 |
| MMEngine | 0.10.7 |
| MMSegmentation | 1.2.2 |
| NumPy | 1.26.4 |
| OpenCV-Python | 4.13.0.92 |
| Pillow | 10.3.0 |

The experiments on the current server were run on a single NVIDIA GeForce RTX 4080 SUPER GPU.

## Installation

```bash
conda create -n hvac-mfr python=3.10 -y
conda activate hvac-mfr

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.2.0

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v1.2.2
pip install -v -e .

pip install numpy==1.26.4 opencv-python==4.13.0.92 pillow==10.3.0 \
  scipy==1.15.3 prettytable==3.18.0 packaging==24.1 yapf==0.43.0
```

## Model registration

The provided configuration files use `custom_imports` to register the HVAC-MFR backbone and decode head automatically. Manual registration is therefore not required for the provided configs. Additional registration notes are provided in `docs/registration.md`.

## Pretraining

For HVAC-MFR, compatible encoder layers can be initialized from ImageNet-1K pretrained MSCAN-T weights, while the newly introduced HVAC and MFRB modules are randomly initialized.

When using a local pretrained checkpoint, set the checkpoint path in the configuration file or override it at runtime:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_1xb4-160k_cityscapes-512x1024.py \
  --cfg-options model.backbone.init_cfg.checkpoint=/path/to/hvac_mfr_in1k_full.pth
```

## Training

PASCAL VOC 2012:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_voc2012-512x512.py
```

Cityscapes:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py
```

Cityscapes with ImageNet-1K compatible initialization:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_1xb4-160k_cityscapes-512x1024.py
```
