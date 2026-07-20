"""Qualitative segmentation figures (boundary quality / thin structures).

Per image, a row:  Image | GT | <model_1> | ... | <model_n> | Boundary-error | Zoom-in
Prediction cells are annotated with Boundary IoU (Cheng et al.) and BF-score;
the zoom-in auto-crops the densest boundary-disagreement region.

Rendering follows MMSegmentation's palette convention; boundary code lives in
``boundary_utils.py`` (adapted from bowenc0221/boundary-iou-api and
minar09/bfscore_python). Models are standard open-source segmentors.

--models entry:  NAME=config.py::checkpoint.pth   (::config optional -> --config)
--compose PATH:  stack all images as rows into ONE figure (paper-ready);
                 otherwise one figure per image is written to --out.
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmseg.apis import init_model, inference_model
import mmseg.models.backbones.hvac_mfr  # noqa: F401
import mmseg.models.decode_heads.hvac_mfr_head  # noqa: F401

from boundary_utils import (mean_boundary_iou, mean_bf_score,
                            semantic_boundary, boundary_error_overlay)


def get_palette(name):
    from mmseg.datasets import PascalVOCDataset, CityscapesDataset
    if name == 'voc':
        return PascalVOCDataset.METAINFO['palette'], 21
    return CityscapesDataset.METAINFO['palette'], 19


def colorize(label, palette):
    out = np.zeros(label.shape + (3,), dtype=np.uint8)
    for idx, col in enumerate(palette):
        out[label == idx] = col
    return out


def overlay(rgb, label, palette, alpha=0.5, keep=(255,)):
    """Blend the colourised label over the RGB image (segmentation-on-photo).

    Pixels whose label is in ``keep`` (e.g. VOC background=0, ignore=255) are
    left as the original photo.
    """
    color = colorize(label, palette).astype(np.float32)
    out = rgb.astype(np.float32).copy()
    m = ~np.isin(label, list(keep))
    out[m] = out[m] * (1 - alpha) + color[m] * alpha
    return out.clip(0, 255).astype(np.uint8)


def predict(model, img_path):
    res = inference_model(model, img_path)
    return res.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)


def resolve_paths(data_root, img_id, dataset):
    if dataset == 'voc':
        return (os.path.join(data_root, 'JPEGImages', img_id + '.jpg'),
                os.path.join(data_root, 'SegmentationClass', img_id + '.png'))
    return (os.path.join(data_root, 'leftImg8bit', 'val', img_id + '_leftImg8bit.png'),
            os.path.join(data_root, 'gtFine', 'val', img_id + '_gtFine_labelTrainIds.png'))


def find_hotspot(err, win_frac=0.35):
    h, w = err.shape
    wh, ww = min(max(int(h * win_frac), 40), h), min(max(int(w * win_frac), 40), w)
    integ = np.pad(err.astype(np.float32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, bx, by = -1.0, 0, 0
    step = max(8, min(wh, ww) // 4)
    for y in range(0, h - wh + 1, step):
        for x in range(0, w - ww + 1, step):
            s = integ[y + wh, x + ww] - integ[y, x + ww] - integ[y + wh, x] + integ[y, x]
            if s > best:
                best, bx, by = s, x, y
    return bx, by, ww, wh


def build_row(img_id, models, bnd_name, data_root, dataset, palette, num_classes):
    """Return list of (header, image, subtext) cells for one image."""
    ip, gp = resolve_paths(data_root, img_id, dataset)
    rgb = np.array(Image.open(ip).convert('RGB'))
    gt = np.array(Image.open(gp))
    preds = {name: predict(m, ip) for name, m in models}
    keep = (0, 255) if dataset == 'voc' else (255,)

    cells = [('Image', rgb, ''), ('GT', overlay(rgb, gt, palette, keep=keep), '')]
    for name, _ in models:
        biou = mean_boundary_iou(gt, preds[name], num_classes)
        bf = mean_bf_score(gt, preds[name], num_classes)
        cells.append((name, overlay(rgb, preds[name], palette, keep=keep),
                      f'BIoU {biou:.3f} | BF {bf:.3f}'))

    be = boundary_error_overlay(rgb, gt, preds[bnd_name], num_classes)
    gt_b = semantic_boundary(gt, num_classes)
    pr_b = semantic_boundary(preds[bnd_name], num_classes)
    x, y, w, h = find_hotspot((gt_b ^ pr_b).astype(np.uint8))
    be_box = be.copy()
    cv2.rectangle(be_box, (x, y), (x + w, y + h), (255, 255, 255), max(2, w // 60))
    cells.append(('Boundary error\n(GT=green pred=red hit=yellow)', be_box, ''))
    cells.append(('Zoom-in (worst boundary)', be[y:y + h, x:x + w], ''))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--img-ids', nargs='+', required=True)
    ap.add_argument('--palette', default='voc', choices=['voc', 'cityscapes'])
    ap.add_argument('--boundary-model', default=None)
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--compose', default=None, help='write ONE stacked figure to this path')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    palette, num_classes = get_palette(args.palette)
    os.makedirs(args.out, exist_ok=True)

    models = []
    for spec in args.models:
        name, rest = spec.split('=', 1)
        cfg, ckpt = (rest.split('::', 1) if '::' in rest else (args.config, rest))
        print(f'[build] {name} <- {cfg} | {ckpt}', flush=True)
        models.append((name, init_model(cfg, ckpt, device=args.device)))
    bnd_name = args.boundary_model or models[-1][0]

    rows = []
    for img_id in args.img_ids:
        rows.append((img_id, build_row(img_id, models, bnd_name, args.data_root,
                                       args.palette, palette, num_classes)))
        print('[done]', img_id, flush=True)

    if args.compose:
        nrow, ncol = len(rows), len(rows[0][1])
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.1 * nrow))
        axes = np.atleast_2d(axes)
        for r, (img_id, cells) in enumerate(rows):
            for c, (header, im, sub) in enumerate(cells):
                ax = axes[r, c]
                ax.imshow(im)
                ax.axis('off')
                if r == 0:
                    ax.set_title(header, fontsize=9)
                if sub:
                    ax.text(0.5, 0.02, sub, transform=ax.transAxes, ha='center',
                            va='bottom', fontsize=7, color='white',
                            bbox=dict(boxstyle='round', fc='black', alpha=0.55, pad=0.2))
        fig.tight_layout()
        fig.savefig(args.compose, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('[composed]', args.compose, flush=True)
    else:
        for img_id, cells in rows:
            n = len(cells)
            fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.6))
            for ax, (header, im, sub) in zip(np.atleast_1d(axes), cells):
                ax.imshow(im)
                ax.set_title(header + (('\n' + sub) if sub else ''), fontsize=8)
                ax.axis('off')
            fig.tight_layout()
            op = os.path.join(args.out, img_id.replace('/', '_') + '_qualitative.png')
            fig.savefig(op, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print('[saved]', op, flush=True)


if __name__ == '__main__':
    main()
