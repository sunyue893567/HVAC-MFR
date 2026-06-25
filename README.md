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

### Resume training

```bash
python tools/train.py configs/hvac_mfr/hvac_mfr-t_1xb4-160k_cityscapes-512x1024.py \
  --work-dir work_dirs/hvac_mfr_cityscapes \
  --resume
```

### Monitor training logs

```bash
tail -f work_dirs/hvac_mfr_cityscapes/*.log
```

To show only training iterations:

```bash
tail -f work_dirs/hvac_mfr_cityscapes/*.log | grep "Iter(train)"
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

The STDC2 variant gives:

| Method | Backbone | Input size | Params (M) | GFLOPs |
|---|---|---|---:|---:|
| HVAC-MFR | STDC2 | 1024 x 2048 | 13.079 | 118.0 |

The mIoU of this variant should be reported after training and validation on the Cityscapes validation set.

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

## Common workflow

```text
1. Install the environment.
2. Prepare the dataset according to the required directory structure.
3. Place HVAC-MFR code and configs under the MMSegmentation project.
4. Train the model with tools/train.py.
5. Monitor logs and checkpoints under work_dirs/.
6. Evaluate the trained checkpoint with tools/test.py.
7. Calculate Params/GFLOPs with tools/analysis_tools/get_flops.py.
8. Run single-image inference with demo/image_demo.py.
```
