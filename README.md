# HVAC-MFR

Lightweight semantic segmentation with horizontal-vertical attention compression and modulated feature refinement.

This repository contains the lightweight HVAC-MFR implementation files and experiment configs designed to be copied into an MMSegmentation project. Dataset files, pretrained weights, checkpoints, work directories, and training logs are intentionally not included.

## Repository contents

- `mmseg/models/backbones/hvac_mfr.py`: HVAC-MFR lightweight encoder.
- `mmseg/models/decode_heads/hvac_mfr_head.py`: HVAC-MFR decode head with GCAM/GSAM refinement.
- `configs/hvac_mfr/`: Cityscapes and PASCAL VOC 2012 training configs.
- `docs/registration.md`: optional permanent registration snippets for MMSegmentation.

## Training environment used in our experiments

The current experiments were run with the following environment:

- OS: Linux x86_64
- Python: 3.10.8
- CUDA runtime used by PyTorch: 11.8
- PyTorch: 2.1.2+cu118
- TorchVision: 0.16.2+cu118
- MMCV: 2.2.0
- MMEngine: 0.10.7
- MMSegmentation: 1.2.2
- NumPy: 1.26.4
- OpenCV-Python: 4.13.0.92
- Pillow: 10.3.0
- GPU used on the current server: NVIDIA GeForce RTX 4080 SUPER, 32GB

## Installation

The following commands reproduce the software environment used on the server. They assume a Linux machine with NVIDIA GPU drivers and Conda/Miniconda installed.

```bash
conda create -n hvac-mfr python=3.10 -y
conda activate hvac-mfr

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.2.0

# Install MMSegmentation 1.2.2.
# If installing from source, use the official MMSegmentation repository.
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v1.2.2
pip install -v -e .

# Additional common dependencies used in the server environment.
pip install numpy==1.26.4 opencv-python==4.13.0.92 pillow==10.3.0 \
  scipy==1.15.3 prettytable==3.18.0 packaging==24.1 yapf==0.43.0
```

## Using HVAC-MFR in MMSegmentation

Copy the files from this repository into the corresponding MMSegmentation directories:

```bash
# Run these commands from the root of an MMSegmentation checkout.
cp -r /path/to/HVAC-MFR/configs/hvac_mfr configs/
cp /path/to/HVAC-MFR/mmseg/models/backbones/hvac_mfr.py mmseg/models/backbones/
cp /path/to/HVAC-MFR/mmseg/models/decode_heads/hvac_mfr_head.py mmseg/models/decode_heads/
```

The provided configs already include `custom_imports` for HVAC-MFR, so manual registration is usually not required. If you prefer permanent registration in MMSegmentation, see `docs/registration.md`.

Prepare datasets following the standard MMSegmentation layout. Large data files are not included in this repository.

## Pretraining note

For HVAC-MFR, compatible encoder layers can be initialized from ImageNet-1K pretrained MSCAN-T weights, while newly introduced HVAC and MFRB modules are randomly initialized. The pretrained weight file is not included in this repository.

If you have the local pretrained checkpoint, set the checkpoint path in the config, for example:

```python
init_cfg=dict(type='Pretrained', checkpoint='pretrain/hvac_mfr_in1k_full.pth')
```

Alternatively, override it at runtime:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_1xb4-160k_cityscapes-512x1024.py \
  --cfg-options model.backbone.init_cfg.checkpoint=/path/to/hvac_mfr_in1k_full.pth
```

## Training examples

PASCAL VOC 2012:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_voc2012-512x512.py
```

Cityscapes without pretrained initialization:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py
```

Cityscapes with ImageNet-1K compatible initialization:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_1xb4-160k_cityscapes-512x1024.py
```

## Not included

The following files are intentionally excluded to keep the repository lightweight:

- datasets such as Cityscapes and PASCAL VOC 2012;
- pretrained weights and checkpoints (`*.pth`, `*.pt`, `*.ckpt`);
- `work_dirs/`, training logs, and temporary experiment outputs.
