# Reproducibility Support

This document clarifies the current public reproducibility support for the
HVAC-MFR repository. It maps the manuscript results that are currently
implemented in the released code to their corresponding configuration files,
checkpoints, and scripts.

At the current stage, the repository provides detailed reproduction support for
the two main comparison results:

- Table 12: PASCAL VOC 2012 comparison result, reproduced with multi-scale/flip
  test-time augmentation (TTA) and an 11-member softmax-probability ensemble.
- Table 13: Cityscapes comparison result, reproduced with the STDC2-transfer
  configuration and TTA.

Other ablation tables and qualitative figures require additional ablation
checkpoints or visualization utilities to be released before they can be fully
reproduced from this repository.

## Code modules

| Component | File |
|---|---|
| HVAC-MFR backbone | `mmseg/models/backbones/hvac_mfr.py` |
| HVAC-MFR decode head | `mmseg/models/decode_heads/hvac_mfr_head.py` |
| VOC ensemble inference | `ensemble_tta_voc.py` |
| VOC ensemble weight search | `ensemble_weight_search.py` |
| One-command VOC reproduction wrapper | `reproduce.sh` |
| VOC ensemble notes | `ENSEMBLE_76.md` |
| VOC experiment table | `EXPERIMENTS_TABLE.md` |

## Reproduction index

| Manuscript item | Dataset | Configuration / code | Checkpoint source | Script | Protocol note |
|---|---|---|---|---|---|
| Table 12 | PASCAL VOC 2012 | `configs/hvac_mfr/hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_tta_voc2012aug.py` | 11 trained ensemble-member checkpoints listed in `reproduce.sh` and `ENSEMBLE_76.md` | `reproduce.sh`, `ensemble_weight_search.py`, `ensemble_tta_voc.py` | 11-member softmax-probability ensemble with multi-scale/flip TTA |
| Table 13 | Cityscapes | `configs/hvac_mfr/hvac_mfr-stdc2-pre-aux_1xb4-160k_cityscapes-512x1024.py` and `configs/hvac_mfr/hvac_mfr-stdc2-pre-aux_tta-dense_cityscapes.py` | STDC2-transfer checkpoint trained from the corresponding configuration | `tools/test.py` with the TTA configuration | STDC2-transfer setting with TTA; the reproduced validation result is approximately 79.2 mIoU |

## Important protocol clarification

The PASCAL VOC 2012 result in Table 12 should be interpreted as an
ensemble/TTA result rather than a single-model single-scale result. The
Cityscapes result in Table 13 should be interpreted as an STDC2-transfer result
with TTA rather than the primary lightweight HVAC-MFR setting.

The scripts and tables in this repository therefore distinguish:

- single-model evaluation;
- test-time augmentation;
- multi-model probability ensemble;
- STDC2-transfer evaluation.

This distinction is important for comparing accuracy, parameter count, GFLOPs,
and deployment-oriented efficiency under consistent experimental protocols.

