# HVAC-MFR

Lightweight semantic segmentation with horizontal-vertical attention compression and modulated feature refinement.

This repository contains the lightweight HVAC-MFR implementation files and experiment configs designed to be copied into an MMSegmentation project.

## Files

- `mmseg/models/backbones/hvac_mfr.py`: HVAC-MFR lightweight encoder.
- `mmseg/models/decode_heads/hvac_mfr_head.py`: HVAC-MFR decode head with GCAM/GSAM refinement.
- `configs/hvac_mfr/`: Cityscapes and PASCAL VOC 2012 training configs.
- `docs/registration.md`: registration snippets for MMSegmentation.

## Usage

1. Prepare an MMSegmentation environment.
2. Copy the files in `mmseg/` and `configs/` into the corresponding MMSegmentation directories.
3. Register the backbone and decode head as described in `docs/registration.md`.
4. Prepare datasets following MMSegmentation conventions.
5. Put pretrained weights locally if needed. Large weights are intentionally not included in this repository.

Example training commands:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_voc2012-512x512.py
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_1xb4-160k_cityscapes-512x1024.py
```

## Pretraining note

For HVAC-MFR, compatible encoder layers can be initialized from ImageNet-1K pretrained MSCAN-T weights, while newly introduced HVAC and MFRB modules are randomly initialized. No dataset files, checkpoints, pretrained weights, work directories, or training logs are included in this repository.
