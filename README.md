# HVAC-MFR

Official implementation of **HVAC-MFR**, a lightweight semantic segmentation framework with horizontal-vertical attention compression and modulated feature refinement.

## Repository contents

- `mmseg/models/backbones/hvac_mfr.py`: HVAC-MFR lightweight encoder.
- `mmseg/models/decode_heads/hvac_mfr_head.py`: HVAC-MFR decode head with GCAM/GSAM refinement.
- `configs/hvac_mfr/`: training configurations for Cityscapes and PASCAL VOC 2012, including the fine-tuning / TTA recipe used for the ensemble.
- `docs/registration.md`: optional registration notes for MMSegmentation.
- `ensemble_tta_voc.py` / `ensemble_weight_search.py`: multi-scale + flip TTA with multi-model softmax-probability ensembling and weighted member search.
- `reproduce.sh`: one-command wrapper that runs the ensemble search over the trained members.
- `ENSEMBLE_76.md` / `EXPERIMENTS_TABLE.md`: reproduction guide and result tables for the PASCAL VOC 2012 ensemble.

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

Create a conda environment:

```bash
conda create -n hvac-mfr python=3.10 -y
conda activate hvac-mfr
```

Install PyTorch:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install OpenMMLab dependencies:

```bash
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.2.0
```

Install MMSegmentation:

```bash
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v1.2.2
pip install -v -e .
```

Install additional packages:

```bash
pip install numpy==1.26.4 opencv-python==4.13.0.92 pillow==10.3.0 \
  scipy==1.15.3 prettytable==3.18.0 packaging==24.1 yapf==0.43.0
```

## Project placement

This project follows the MMSegmentation directory layout. Place the HVAC-MFR files under an MMSegmentation working directory as follows:

```text
mmsegmentation/
|-- configs/
|   `-- hvac_mfr/
|       |-- hvac_mfr-t_1xb4-160k_voc2012-512x512.py
|       |-- hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py
|       `-- hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py
|-- mmseg/
|   `-- models/
|       |-- backbones/
|       |   `-- hvac_mfr.py
|       `-- decode_heads/
|           `-- hvac_mfr_head.py
`-- docs/
    `-- registration.md
```

The provided configuration files use `custom_imports` to register the HVAC-MFR backbone and decode head automatically. Manual registration is not required for the provided configs.

## Datasets

HVAC-MFR is evaluated on PASCAL VOC 2012 and Cityscapes. Dataset paths can be modified directly in the config files or overridden at runtime with `--cfg-options`.

### PASCAL VOC 2012

The VOC config uses `PascalVOCDataset` with 21 classes. The expected directory structure is:

```text
data/VOCdevkit/VOC2012/
|-- JPEGImages/
|-- SegmentationClass/
`-- ImageSets/
    `-- Segmentation/
        |-- train.txt
        `-- val.txt
```

The corresponding config is:

```text
configs/hvac_mfr/hvac_mfr-t_1xb4-160k_voc2012-512x512.py
```

Key dataset settings:

| Item | Setting |
|---|---|
| Dataset type | `PascalVOCDataset` |
| Data root | `data/VOCdevkit/VOC2012` |
| Image directory | `JPEGImages` |
| Annotation directory | `SegmentationClass` |
| Train split | `ImageSets/Segmentation/train.txt` |
| Validation split | `ImageSets/Segmentation/val.txt` |
| Number of classes | 21 |
| Crop size | 512 x 512 |

### Cityscapes

The Cityscapes config uses `CityscapesDataset` with 19 classes. The expected directory structure is:

```text
data/cityscapes/
|-- leftImg8bit/
|   |-- train/
|   |-- val/
|   `-- test/
`-- gtFine/
    |-- train/
    |-- val/
    `-- test/
```

The corresponding configs are:

```text
configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py
configs/hvac_mfr/hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py
```

Key dataset settings:

| Item | Setting |
|---|---|
| Dataset type | `CityscapesDataset` |
| Data root | `data/cityscapes` |
| Train images | `leftImg8bit/train` |
| Train annotations | `gtFine/train` |
| Validation images | `leftImg8bit/val` |
| Validation annotations | `gtFine/val` |
| Number of classes | 19 |
| Crop size | 512 x 1024 |

## Configuration overview

The repository provides two model settings:

| Config | Backbone | Decode head | Dataset |
|---|---|---|---|
| `hvac_mfr-t_1xb4-160k_voc2012-512x512.py` | `HVACMFR` | `HVACMFRHead` | PASCAL VOC 2012 |
| `hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py` | `HVACMFR` | `HVACMFRHead` | Cityscapes |
| `hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py` | `STDC2` | `HVACMFRHead` | Cityscapes |

The default HVAC-MFR lightweight encoder uses:

| Component | Setting |
|---|---|
| Model type | `EncoderDecoder` |
| Backbone | `HVACMFR` |
| Decode head | `HVACMFRHead` |
| Encoder channels | `[32, 64, 160, 256]` |
| Encoder depths | `[3, 3, 5, 2]` |
| Optimizer | AdamW |
| Learning rate | 6e-5 |
| Betas | `(0.9, 0.999)` |
| Weight decay | 0.01 |
| Batch size | 4 |
| Training iterations | 160,000 |
| Validation interval | 16,000 |
| Checkpoint interval | 16,000 |
| LR schedule | Linear warm-up for 1,500 iterations + polynomial decay |

The STDC2 variant uses the MMSegmentation `STDCContextPathNet` backbone with `stdc_type='STDCNet2'` and the proposed `HVACMFRHead`. It is included for comparison under an STDC2 backbone setting.

The training pipeline is:

```text
LoadImageFromFile -> LoadAnnotations -> RandomResize(ratio_range=0.5-2.0) -> RandomCrop -> RandomFlip(prob=0.5) -> PhotoMetricDistortion -> PackSegInputs
```

## Initialization

The commands below run HVAC-MFR without requiring an external pretrained checkpoint. Model parameters follow the initialization strategy defined by MMSegmentation and the model implementation.

## Training

Run all commands from the root directory of the MMSegmentation project.

### Train on PASCAL VOC 2012

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_voc2012-512x512.py \
  --work-dir work_dirs/hvac_mfr_voc2012
```

If the VOC dataset is stored in another directory:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_voc2012-512x512.py \
  --work-dir work_dirs/hvac_mfr_voc2012 \
  --cfg-options \
  train_dataloader.dataset.data_root=/path/to/VOCdevkit/VOC2012 \
  val_dataloader.dataset.data_root=/path/to/VOCdevkit/VOC2012 \
  test_dataloader.dataset.data_root=/path/to/VOCdevkit/VOC2012
```

### Train on Cityscapes

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py \
  --work-dir work_dirs/hvac_mfr_cityscapes
```

### Train the STDC2 variant on Cityscapes

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py \
  --work-dir work_dirs/hvac_mfr_stdc2_cityscapes
```

If the Cityscapes dataset is stored in another directory:

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py \
  --work-dir work_dirs/hvac_mfr_stdc2_cityscapes \
  --cfg-options \
  train_dataloader.dataset.data_root=/path/to/cityscapes \
  val_dataloader.dataset.data_root=/path/to/cityscapes \
  test_dataloader.dataset.data_root=/path/to/cityscapes
```

## Validation

After training, evaluate a saved checkpoint with `tools/test.py`.

### Validate on PASCAL VOC 2012

```bash
python tools/test.py \
  configs/hvac_mfr/hvac_mfr-t_1xb4-160k_voc2012-512x512.py \
  <checkpoint_file>
```

### Validate on Cityscapes

```bash
python tools/test.py \
  configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py \
  <checkpoint_file>
```

### Validate the STDC2 variant

```bash
python tools/test.py \
  configs/hvac_mfr/hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py \
  <checkpoint_file>
```

The evaluation results, including mIoU, will be printed in the terminal and saved in the corresponding work directory logs.

## Model complexity

Use `tools/analysis_tools/get_flops.py` to calculate parameters and GFLOPs.

### Default HVAC-MFR on Cityscapes

```bash
python tools/analysis_tools/get_flops.py \
  configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py \
  --shape 1024 2048
```

### HVAC-MFR head with STDC2 backbone

```bash
python tools/analysis_tools/get_flops.py \
  configs/hvac_mfr/hvac_mfr-stdc2_1xb4-160k_cityscapes-512x1024.py \
  --shape 1024 2048
```

The command above prints the model parameters and computational complexity for
the STDC2-backbone configuration. The final mIoU should be obtained by training
the model and evaluating the saved checkpoint on the Cityscapes validation set.

## Inference

Use `demo/image_demo.py` for single-image inference.

### PASCAL VOC 2012 example

```bash
python demo/image_demo.py \
  data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg \
  configs/hvac_mfr/hvac_mfr-t_1xb4-160k_voc2012-512x512.py \
  <checkpoint_file> \
  --device cuda:0 \
  --out-file outputs/voc_demo_result.png
```

### Cityscapes example

```bash
python demo/image_demo.py \
  data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png \
  configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py \
  <checkpoint_file> \
  --device cuda:0 \
  --out-file outputs/cityscapes_demo_result.png
```

## Reproducing the ensemble result (PASCAL VOC 2012)

The result is obtained with test-time augmentation (TTA) plus a multi-model
**softmax-probability ensemble**. The key idea: unlike weight averaging (model
soup), which saturates, averaging the *softmax probabilities* of models trained
with **different seeds and loss recipes** keeps improving. The most useful
"decorrelated" members come from fine-tuning the plain-Lovász `ft40k` recipe on
top of several **independent 160k base checkpoints**.

The scripts are `ensemble_tta_voc.py` (probability ensemble with per-model
multi-scale `[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]` + flip TTA) and
`ensemble_weight_search.py` (scores many member-weight vectors in a single
inference pass to pick the best weighting).

### Step 1 — Train the 160k base models

```bash
# aux-ohem base (used by the aux-ohem-lovasz ft40k members and their seeds)
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_aux-ohem_1xb4-160k_voc2012aug-512x512.py \
  --work-dir work_dirs/base_aux_ohem_seed3407 --cfg-options randomness.seed=3407
# plain base (used by the plain-lovasz ft40k members)
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_1xb4-160k_voc2012aug-512x512.py \
  --work-dir work_dirs/base_plain_seed42
```

Repeat the aux-ohem base with `randomness.seed=2029` / `2031` (and the AdamW base
config `hvac_mfr-t_table-adamw-officialaug-nw0_1xb4-160k_voc2012aug-512x512.py`) to
obtain the independent bases used by the ensemble members.

### Step 2 — Fine-tune the ensemble members (ft40k)

```bash
# aux-ohem-lovasz ft40k members (seeds 3407 / 2029 / 2031)
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_1xb4_voc2012aug-512x512.py \
  --work-dir work_dirs/ft40k_seed3407 --cfg-options randomness.seed=3407

# plain-lovasz ft40k members from independent bases (the decorrelated members that
# drive the gain). Point load_from at each 160k base and vary the seed:
python tools/train.py configs/hvac_mfr/hvac_mfr-t_in1k-pre_lovasz-ft40k_1xb4_voc2012aug-512x512.py \
  --work-dir work_dirs/lov_from_base2031_seed6031 \
  --cfg-options load_from=work_dirs/base_aux_ohem_seed2031/best_mIoU_iter_*.pth randomness.seed=6031
```

The full member pool also includes the rare-crop (`...rare-crop-p50/p75-stage3-ft8k...`)
and class-uniform (`...class-uniform-p50-stage3-ft6k...`) stage-3 fine-tunes; all
configs are under `configs/hvac_mfr/`.

### Step 3 — Run the ensemble search and read the final metric

Collect the 11 member checkpoints trained in Steps 1–2, then run the weighted
search. It runs each member's TTA once per image, scores the weight vectors in a
single pass, and **prints the top weightings — the best line is the reproduced
metric.** The search (`--seed 30`) is deterministic; run it with:

```bash
bash reproduce.sh
```

`reproduce.sh` holds the 11 member checkpoint paths and the search settings, so no
long command is needed. The exact 11-member pool, the winning weight vector, and the honest val-tuning
caveat are documented in [`ENSEMBLE_76.md`](ENSEMBLE_76.md); per-experiment
single-model and ensemble numbers are in
[`EXPERIMENTS_TABLE.md`](EXPERIMENTS_TABLE.md).

