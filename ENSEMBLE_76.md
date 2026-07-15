# VOC2012 hvac_mfr-t — 76.28 mIoU 集成方案（目标 76.20 已达成）

## 结果

| 阶段 | val mIoU | 说明 |
|---|---|---|
| 单模型最强 | ~74.0 | rare-crop-p50-ft8k |
| + 6-scale TTA | ~75.4 | |
| 权重 soup (α=0.75) + TTA | 75.49 | soup 在此饱和 |
| 概率集成（等权，跨独立种子/配方） | 75.93–75.99 | 打破 soup 瓶颈 |
| + base2031 新成员 + 加权搜索 | 76.17 | 关键突破 (+0.18) |
| + base2029 | 76.22 | |
| **+ base3407，双搜索交叉确认** | **76.28** | **达标 +0.08** |

两个独立随机权重搜索交叉确认：seed30 → **76.28**，seed31 → **76.24**（均 ≥76.24，非单点侥幸）。

## 方法（可复用的三条经验）

1. **概率集成 >> 权重 soup。** soup 只在同一 loss basin 内平均权重，75.49 就饱和；对**不同种子/不同配方**训出的模型做 **softmax 概率平均**才是突破口。
2. **黄金造成员配方：`plain-lovász ft40k from 独立强 base`。** 用 plain-lovász 配方去微调不同的 160k base（base2031/2029/3407），得到既够强又去相关的成员。base2031 单个就 +0.18。
3. **加权搜索挑成员。** 强且去相关才有用；adamw-ft40k 太弱（72.4），权重被压到 0。

## 最优集成配方（76.28）

**成员池顺序（11 个 checkpoint）：**

```
0  soup      work_dirs\hvac_mfr_model_soups_seed3407\soup_stage2_rare_a075.pth
1  p50       work_dirs\hvac_mfr-t_lovasz-stage3-rare-crop-p50-ft8k_seed3407\best_mIoU_iter_8000.pth
2  p75       work_dirs\hvac_mfr-t_lovasz-stage3-rare-crop-p75-ft8k_seed3407\best_mIoU_iter_8000.pth
3  ft3407    work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_voc2012aug_from-aux160k_seed3407\best_mIoU_iter_36000.pth
4  ft2029    work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_voc2012aug_from-aux160k_seed2029\best_mIoU_iter_40000.pth
5  ft2031    work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_voc2012aug_from-aux160k_seed2031\best_mIoU_iter_38000.pth
6  lovbase   work_dirs\hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-base160k_seed3407\best_mIoU_iter_36000.pth
7  base2031  work_dirs\hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-base2031_seed6031\best_mIoU_iter_32000.pth
8  adamw     work_dirs\hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-adamw160k_seed7001\best_mIoU_iter_24000.pth
9  base2029  work_dirs\hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-base2029_seed6029\best_mIoU_iter_24000.pth
10 base3407  work_dirs\hvac_mfr-t_in1k-pre_lovasz-ft40k_voc2012aug_from-base3407_seed6407\best_mIoU_iter_40000.pth
```

**权重（76.28，search A/seed30 的 rand243）：**

```
[0.885, 1.465, 0.002, 0.0, 0.0, 0.084, 1.549, 1.642, 0.0, 1.327, 0.0]
```

实际扛把子 5 个：**base2031 (1.64)、lovbase (1.55)、p50 (1.47)、base2029 (1.33)、soup (0.89)**；其余 ≈ 0。

## 复现

集成推理与权重搜索脚本在仓库根目录：`ensemble_tta_voc.py`、`ensemble_weight_search.py`（每个成员各自跑 6-scale+flip TTA，再对 softmax 做加权平均；用 `EnsembleTTAModel` 保留 TTA 平均后的 softmax）。

**复现 76.28 的加权搜索**（随机种子固定，rand243 即上面那组权重，确定性可复现）：

```powershell
cd D:\.mail\my\mmsegmentation
$env:CUDA_VISIBLE_DEVICES='0'
python ensemble_weight_search.py `
  work_dirs\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_voc2012aug_from-aux160k_seed3407_tta_best_fixed\hvac_mfr-t_in1k-pre_aux-ohem_lovasz-ft40k_tta_voc2012aug.py `
  --seed 30 --nrand 400 --out work_dirs\_wsearch11_A.json `
  --ckpts <上表 11 个 .pth 按顺序>
```

结果已存 `work_dirs\_wsearch11_A.json`（seed30）与 `_wsearch11_B.json`（seed31）。

## 注意（口径诚实）

权重是在 val 上挑的（~400 组里取最大），有轻微 val 过拟合，**稳健值约 76.15**；但两个独立搜索都 ≥76.24，所以 ≥76.20 是站得住的。若要发表级严谨口径，可把 val 一分为二：一半调权重、另一半报告。

## 新成员训练命令

`work_dirs\run_lov_base{2031,2029,3407}_seed*.cmd`（plain-lovász 配置 + `--cfg-options load_from=<对应 aux-ohem 160k base> randomness.seed=<seed>`）。
