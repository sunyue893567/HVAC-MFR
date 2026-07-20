"""Boundary utilities for qualitative segmentation figures.

Adapted (with attribution) from established open-source implementations:

* Boundary IoU API - Bowen Cheng et al., "Boundary IoU: Improving Object-Centric
  Image Segmentation Evaluation", CVPR 2021.
  https://github.com/bowenc0221/boundary-iou-api
  -> ``mask_to_boundary`` is taken verbatim from
     ``boundary_iou/utils/boundary_utils.py``.

* bfscore_python - https://github.com/minar09/bfscore_python
  -> the contour/boundary F-score (BF-score) idea; here implemented with the
     standard dilate-and-match formulation.
"""
import cv2
import numpy as np


# ---- from boundary-iou-api (Bowen Cheng), verbatim -------------------------
def mask_to_boundary(mask, dilation_ratio=0.02):
    """Convert a binary mask to its boundary mask.

    boundary = mask - erode(mask, dilation), with
    dilation = round(dilation_ratio * image_diagonal).
    """
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad so a mask touching the border still yields a boundary there.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1:h + 1, 1:w + 1]
    return mask - mask_erode


# ---- Boundary IoU (Cheng et al.) ------------------------------------------
def boundary_iou(gt_mask, dt_mask, dilation_ratio=0.02):
    """Boundary IoU for a single binary (foreground) mask pair."""
    gt_b = mask_to_boundary(gt_mask, dilation_ratio)
    dt_b = mask_to_boundary(dt_mask, dilation_ratio)
    inter = np.logical_and(gt_b, dt_b).sum()
    union = np.logical_or(gt_b, dt_b).sum()
    return float(inter / union) if union > 0 else 1.0


def semantic_boundary(label, num_classes, dilation_ratio=0.02, ignore_index=255):
    """Union of per-class boundaries of a multi-class label map -> binary map."""
    bnd = np.zeros(label.shape, dtype=np.uint8)
    for c in range(num_classes):
        m = (label == c).astype(np.uint8)
        if m.sum() == 0:
            continue
        bnd |= mask_to_boundary(m, dilation_ratio)
    return (bnd > 0).astype(np.uint8)


def mean_boundary_iou(gt, pred, num_classes, dilation_ratio=0.02, ignore_index=255):
    """Mean Boundary IoU over the classes that appear in ``gt``."""
    ious = []
    for c in range(num_classes):
        g = (gt == c).astype(np.uint8)
        if g.sum() == 0:
            continue
        ious.append(boundary_iou(g, (pred == c).astype(np.uint8), dilation_ratio))
    return float(np.mean(ious)) if ious else float('nan')


# ---- Boundary F-score (BF-score), dilate-and-match ------------------------
def bf_score(gt_mask, dt_mask, theta=2, dilation_ratio=0.02):
    """Boundary F1 score for a binary mask pair.

    precision = |pred_boundary within theta of gt_boundary| / |pred_boundary|
    recall    = |gt_boundary   within theta of pred_boundary| / |gt_boundary|
    """
    gt_b = mask_to_boundary(gt_mask, dilation_ratio).astype(np.uint8)
    dt_b = mask_to_boundary(dt_mask, dilation_ratio).astype(np.uint8)
    if gt_b.sum() == 0 and dt_b.sum() == 0:
        return 1.0
    k = 2 * int(theta) + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    gt_b_d = cv2.dilate(gt_b, kernel)
    dt_b_d = cv2.dilate(dt_b, kernel)
    prec = (dt_b & gt_b_d).sum() / dt_b.sum() if dt_b.sum() > 0 else 0.0
    rec = (gt_b & dt_b_d).sum() / gt_b.sum() if gt_b.sum() > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def mean_bf_score(gt, pred, num_classes, theta=2, dilation_ratio=0.02, ignore_index=255):
    scores = []
    for c in range(num_classes):
        g = (gt == c).astype(np.uint8)
        if g.sum() == 0:
            continue
        scores.append(bf_score(g, (pred == c).astype(np.uint8), theta, dilation_ratio))
    return float(np.mean(scores)) if scores else float('nan')


def boundary_error_overlay(rgb, gt, pred, num_classes, dilation_ratio=0.02,
                           alpha=0.9):
    """Colour the semantic boundaries on top of the RGB image.

    green  = GT boundary only (missed)
    red    = prediction boundary only (false boundary)
    yellow = boundary agreement (GT and prediction overlap)
    """
    gt_b = semantic_boundary(gt, num_classes, dilation_ratio)
    pr_b = semantic_boundary(pred, num_classes, dilation_ratio)
    out = rgb.astype(np.float32).copy()
    both = (gt_b & pr_b) > 0
    gt_only = (gt_b > 0) & (~both)
    pr_only = (pr_b > 0) & (~both)
    # colours are RGB
    for m, col in ((gt_only, (0, 255, 0)), (pr_only, (255, 0, 0)), (both, (255, 255, 0))):
        for ch in range(3):
            out[..., ch][m] = (1 - alpha) * out[..., ch][m] + alpha * col[ch]
    return out.clip(0, 255).astype(np.uint8)
