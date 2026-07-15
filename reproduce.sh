#!/usr/bin/env bash
# Reproduce the PASCAL VOC 2012 ensemble result.
# Run from the MMSegmentation project root, after training the members (README Steps 1-2).
# It runs the weighted probability-ensemble search and prints the best mIoU.
set -e

CFG=configs/hvac_mfr/hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_tta_voc2012aug.py

# The 11 ensemble members (best_mIoU checkpoints produced in Steps 1-2).
CKPTS=(
  work_dirs/hvac_mfr_model_soups_seed3407/soup_stage2_rare_a075.pth
  work_dirs/hvac_mfr-t_lovasz-stage3-rare-crop-p50-ft8k_seed3407/best_mIoU_iter_8000.pth
  work_dirs/hvac_mfr-t_lovasz-stage3-rare-crop-p75-ft8k_seed3407/best_mIoU_iter_8000.pth
  work_dirs/hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_voc2012aug_from-aux160k_seed3407/best_mIoU_iter_36000.pth
  work_dirs/hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_voc2012aug_from-aux160k_seed2029/best_mIoU_iter_40000.pth
  work_dirs/hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_voc2012aug_from-aux160k_seed2031/best_mIoU_iter_38000.pth
  work_dirs/hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-base160k_seed3407/best_mIoU_iter_36000.pth
  work_dirs/hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-base2031_seed6031/best_mIoU_iter_32000.pth
  work_dirs/hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-adamw160k_seed7001/best_mIoU_iter_24000.pth
  work_dirs/hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-base2029_seed6029/best_mIoU_iter_24000.pth
  work_dirs/hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-base3407_seed6407/best_mIoU_iter_40000.pth
)

python ensemble_weight_search.py "$CFG" --seed 30 --nrand 400 --ckpts "${CKPTS[@]}"
