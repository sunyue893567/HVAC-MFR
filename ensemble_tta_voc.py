# Multi-model softmax-probability ensemble with per-model multi-scale + flip TTA.
# All members share the same mfr-t architecture (VOC, 21 classes); only weights differ.
# Each model runs the config's TTA pipeline (SegTTAModel averages softmax over the
# augmented views); we then average those per-model softmax maps and take argmax.
import argparse
import copy

import torch
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.structures import PixelData

from mmseg.registry import MODELS

# register custom backbone / head so configs build
import mmseg.models.backbones.hvac_mfr  # noqa: F401
import mmseg.models.decode_heads.hvac_mfr_head  # noqa: F401
from mmseg.models.segmentors.seg_tta import SegTTAModel


@MODELS.register_module()
class EnsembleTTAModel(SegTTAModel):
    """SegTTAModel that also keeps the TTA-averaged softmax in ``seg_logits``
    so it can be averaged across several models."""

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
            # keep the averaged probabilities for cross-model ensembling
            data_sample.set_data({'seg_logits': PixelData(data=logits)})
            if self.module.out_channels == 1:
                seg_pred = (logits > self.module.decode_head.threshold).to(
                    logits).squeeze(1)
            else:
                seg_pred = logits.argmax(dim=0)
            data_sample.set_data({'pred_sem_seg': PixelData(data=seg_pred)})
            if hasattr(data_samples[0], 'gt_sem_seg'):
                data_sample.set_data(
                    {'gt_sem_seg': data_samples[0].gt_sem_seg})
            data_sample.set_metainfo({'img_path': data_samples[0].img_path})
            predictions.append(data_sample)
        return predictions


def build_model(cfg, ckpt, device):
    model = MODELS.build(cfg.tta_model)
    load_checkpoint(model.module, ckpt, map_location='cpu', strict=False)
    model.eval()
    model.to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='any member TTA config (shared arch + val pipeline)')
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    cfg.tta_model.type = 'EnsembleTTAModel'

    # keep host-RAM low so several ensembles can run concurrently
    cfg.test_dataloader.num_workers = 2
    cfg.test_dataloader.persistent_workers = False

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    evaluator = Evaluator(cfg.test_evaluator)
    evaluator.dataset_meta = dataloader.dataset.metainfo

    device = args.device
    models = [build_model(cfg, c, device) for c in args.ckpts]
    print(f'[ensemble] loaded {len(models)} models on {device}', flush=True)
    for c in args.ckpts:
        print('   -', c, flush=True)

    n = len(dataloader.dataset)
    for idx, data_batch in enumerate(dataloader):
        acc = None
        gt = None
        with torch.no_grad():
            for m in models:
                # each model's averaged-softmax is consumed immediately below,
                # before the next model overwrites the shared data_samples, so no
                # copy of the (large multi-scale) batch is needed.
                out = m.test_step(data_batch)
                ds = out[0]
                sl = ds.seg_logits.data
                acc = sl.clone() if acc is None else acc + sl
                if gt is None and hasattr(ds, 'gt_sem_seg'):
                    gt = ds.gt_sem_seg.data.clone()
        pred = acc.argmax(dim=0)
        sample = {
            'pred_sem_seg': {'data': pred},
            'gt_sem_seg': {'data': gt},
        }
        evaluator.process(data_samples=[sample], data_batch=data_batch)
        if (idx + 1) % 200 == 0:
            print(f'[ensemble] {idx + 1}/{n}', flush=True)

    metrics = evaluator.evaluate(n)
    print('=' * 60, flush=True)
    print('ENSEMBLE RESULT:', metrics, flush=True)
    print('=' * 60, flush=True)


if __name__ == '__main__':
    main()
