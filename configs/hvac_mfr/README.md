# HVAC-MFR Configurations

This folder contains the MMSegmentation configs for HVAC-MFR.

## Config list

| Config | Backbone | Decode head | Dataset | Purpose |
|---|---|---|---|---|
| `hvac_mfr-t_1xb4-160k_voc2012-512x512.py` | HVAC-MFR lightweight encoder | HVACMFRHead | PASCAL VOC 2012 | Main VOC experiment |
| `hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py` | HVAC-MFR lightweight encoder | HVACMFRHead | Cityscapes | Main Cityscapes experiment |
| `hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py` | STDC2 | HVACMFRHead | Cityscapes | STDC2-backbone comparison |

## Train examples

Run from the root of an MMSegmentation checkout.

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py \
  --work-dir work_dirs/hvac_mfr_cityscapes
```

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py \
  --work-dir work_dirs/hvac_mfr_stdc2_cityscapes
```

## Validate examples

```bash
python tools/test.py \
  configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py \
  <checkpoint_file>
```

```bash
python tools/test.py \
  configs/hvac_mfr/hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py \
  <checkpoint_file>
```

## Complexity analysis

For the STDC2 variant:

```bash
python tools/analysis_tools/get_flops.py \
  configs/hvac_mfr/hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py \
  --shape 1024 2048
```

Expected complexity:

| Method | Year | Backbone | Params (M) | GFLOPs | mIoU (%) |
|---|---:|---|---:|---:|---:|
| HVAC-MFR | - | STDC2 | 13.1 | 117.8 | 79.2 |

The values are reported with one decimal place to match the comparison table format.
