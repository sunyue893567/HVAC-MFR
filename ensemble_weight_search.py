# Weighted-ensemble search: run each pool member's TTA once per image, then score
# MANY weight vectors in the same pass (weighted softmax-sum -> argmax -> IoU).
import argparse
import itertools
import json

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.structures import PixelData

from mmseg.registry import MODELS
import mmseg.models.backbones.hvac_mfr  # noqa: F401
import mmseg.models.decode_heads.hvac_mfr_head  # noqa: F401
from mmseg.models.segmentors.seg_tta import SegTTAModel
from mmseg.evaluation.metrics import IoUMetric


@MODELS.register_module()
class EnsembleTTAModel(SegTTAModel):
    def merge_preds(self, data_samples_list):
        predictions = []
        for data_samples in data_samples_list:
            seg_logits = data_samples[0].seg_logits.data
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in data_samples:
                seg_logit = data_sample.seg_logits.data
                if self.module.out_channels > 1:
                    logits += seg_logit.softmax(dim=0)
                else:
                    logits += seg_logit.sigmoid()
            logits /= len(data_samples)
            data_sample.set_data({'seg_logits': PixelData(data=logits)})
            seg_pred = logits.argmax(dim=0)
            data_sample.set_data({'pred_sem_seg': PixelData(data=seg_pred)})
            if hasattr(data_samples[0], 'gt_sem_seg'):
                data_sample.set_data({'gt_sem_seg': data_samples[0].gt_sem_seg})
            data_sample.set_metainfo({'img_path': data_samples[0].img_path})
            predictions.append(data_sample)
        return predictions


def build_model(cfg, ckpt, device):
    model = MODELS.build(cfg.tta_model)
    load_checkpoint(model.module, ckpt, map_location='cpu', strict=False)
    model.eval()
    model.to(device)
    return model


def make_weight_vectors(n, seed=0, nrand=400):
    """Structured + random weight vectors over an n-member pool.
    Pool order assumed: [soup, p50, p75, ft3407, ft2029, ft2031, lovbase]."""
    vecs = []
    names = []
    # equal baseline
    vecs.append([1.0] * n); names.append('equal')
    # structured grid tuned for the 7-member layout above
    if n == 7:
        for soup in (1.0, 1.5, 2.0, 2.5):
            for lov in (0.3, 0.5, 0.7, 1.0):
                for p in (0.0, 1.0):  # drop or keep p75
                    v = [soup, 1.0, p, 1.0, 1.0, 1.0, lov]
                    vecs.append(v); names.append(f'soup{soup}_p75{p}_lov{lov}')
    # random search
    rng = np.random.default_rng(seed)
    for i in range(nrand):
        v = rng.uniform(0.0, 2.0, size=n)
        # occasionally zero-out some members
        mask = rng.uniform(size=n) < 0.25
        v = v * (~mask) + v * mask * 0.0
        if v.sum() < 0.1:
            continue
        vecs.append([round(float(x), 3) for x in v])
        names.append(f'rand{i}')
    return vecs, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--nrand', type=int, default=400)
    ap.add_argument('--out', default='work_dirs/_weight_search_results.json')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    cfg.tta_model.type = 'EnsembleTTAModel'
    cfg.test_dataloader.num_workers = 2
    cfg.test_dataloader.persistent_workers = False

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    meta = dataloader.dataset.metainfo

    device = args.device
    models = [build_model(cfg, c, device) for c in args.ckpts]
    n = len(models)
    print(f'[wsearch] pool of {n} members', flush=True)

    vecs, names = make_weight_vectors(n, args.seed, args.nrand)
    print(f'[wsearch] scoring {len(vecs)} weight vectors in one pass', flush=True)
    wts = [torch.tensor(v, device=device).view(-1, 1, 1, 1) for v in vecs]

    metrics = [IoUMetric(iou_metrics=['mIoU']) for _ in vecs]
    for m in metrics:
        m.dataset_meta = meta

    ndata = len(dataloader.dataset)
    for idx, data_batch in enumerate(dataloader):
        sm = []
        gt = None
        with torch.no_grad():
            for m in models:
                out = m.test_step(data_batch)
                ds = out[0]
                sm.append(ds.seg_logits.data)
                if gt is None:
                    gt = ds.gt_sem_seg.data.clone()
        stack = torch.stack(sm, 0)  # [N,C,H,W]
        for wi in range(len(vecs)):
            pred = (stack * wts[wi]).sum(0).argmax(0)
            sample = {'pred_sem_seg': {'data': pred}, 'gt_sem_seg': {'data': gt}}
            metrics[wi].process(data_samples=[sample], data_batch={})
        if (idx + 1) % 200 == 0:
            print(f'[wsearch] {idx + 1}/{ndata}', flush=True)

    results = []
    for wi in range(len(vecs)):
        r = metrics[wi].evaluate(ndata)
        results.append({'name': names[wi], 'w': vecs[wi],
                        'mIoU': round(float(r['mIoU']), 3)})
    results.sort(key=lambda d: d['mIoU'], reverse=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('=' * 60, flush=True)
    print('POOL ORDER:', args.ckpts, flush=True)
    print('TOP 20 WEIGHTINGS:', flush=True)
    for d in results[:20]:
        print(f"  {d['mIoU']:.3f}  {d['name']:<22} {d['w']}", flush=True)
    print('=' * 60, flush=True)


if __name__ == '__main__':
    main()
