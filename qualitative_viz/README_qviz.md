# Qualitative visualization module (reviewer response)

Generates the qualitative figures for boundary quality / thin structures / small
objects, per image:

```
Image | GT | model_1 | ... | model_n | Boundary-error | Zoom-in
```

Each prediction panel is annotated with **Boundary IoU** and **BF-score** vs GT.
The boundary-error panel colours: **green** = GT boundary missed, **red** = false
predicted boundary, **yellow** = boundary agreement.

## Provenance (adapted, not our own code)

- `boundary_utils.py`
  - `mask_to_boundary`: verbatim from **boundary-iou-api** (Bowen Cheng et al.,
    *Boundary IoU*, CVPR 2021), https://github.com/bowenc0221/boundary-iou-api
  - BF-score: dilate-and-match formulation, after
    **bfscore_python** (https://github.com/minar09/bfscore_python)
- `visualize.py` rendering follows MMSegmentation's `SegLocalVisualizer` palette
  convention (https://github.com/open-mmlab/mmsegmentation).

## Usage

Run from the MMSegmentation project root. Fill in your own ablation checkpoints
in panel order (`NAME=path.pth`):

```bash
python qualitative_viz/visualize.py \
  --config configs/hvac_mfr/hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_1xb4_voc2012aug-512x512.py \
  --data-root D:/.mail/my/Code/s2_corr-main/data/VOCdevkit/VOC2012 \
  --palette voc \
  --img-ids 2007_000033 2007_000042 2007_000123 \
  --checkpoints \
     Baseline=work_dirs/<baseline>/best_mIoU_iter_*.pth \
     "+HVAC=work_dirs/<hvac>/best_mIoU_iter_*.pth" \
     "+MFRB=work_dirs/<mfrb>/best_mIoU_iter_*.pth" \
     HVAC-MFR=work_dirs/<full>/best_mIoU_iter_*.pth \
  --zoom 120,80,160,160 \
  --out qualitative_viz/outputs
```

For Cityscapes: `--palette cityscapes` and point `--data-root`/`--img-ids` at the
Cityscapes val images (edit the image/GT sub-paths in `visualize.py` if needed).

## Notes

- `--boundary-model NAME` chooses which model's prediction drives the
  boundary-error panel (default: the last one, i.e. HVAC-MFR).
- Recommended VOC classes for thin structures / small objects: bicycle, bottle,
  chair, person, sofa, pottedplant. Cityscapes: pole, traffic light/sign, rider,
  bicycle, motorcycle, person edges.
