# ICRA-MIRNet: Low-Light Industrial Enhancement & Anomaly Detection

This repository tracks a reproducible study on low-light industrial inspection. It contains a
faithful reproduction of **MIRNet-v2** (*Learning Enriched Features for Fast Image Restoration
and Enhancement*, IEEE TPAMI 2022) and our illumination-conditioned residual adapter,
**ICRA-MIRNet**, evaluated on **MVTec AD (15 classes)** and **MPDD** anomaly detection.

## Key Results

Three full random seeds (20260830-20260832), MVTec AD 15 classes, 1,725 test images,
12,000 micro-batches (gradient accumulation 4 = 3,000 optimizer updates), mean +/- sample std:

| Metric | Value |
|---|---|
| PSNR | 28.472 +/- 0.243 dB |
| SSIM | 0.8946 +/- 0.0015 |
| Clean AUROC | 0.8211 +/- 0.0000 |
| Synthetic low-light AUROC | 0.6712 +/- 0.0050 |
| Restored AUROC | 0.7378 +/- 0.0066 |

Restoration improves AUROC by 0.0666 on average over the low-light input, but remains 0.0833
below the clean-image baseline. Enhancement does **not** always improve detection: the pre-registered
mild condition drops restored AUROC (0.7400 -> 0.7119), severe improves it (0.6326 -> 0.6584), and
zero-shot transfer from MVTec to MPDD degrades it (0.6435 -> 0.4777). All low-light images are
programmatically synthesized; no claim is made about real dark-factory deployment.

## Repository Layout

```text
MIRNetv2/                        Pinned source (upstream swz30/MIRNetv2) + our industrial additions
  industrial/                    ICRA-MIRNet adapter, runner, and experiment configs
  basicsr/                       Model architecture, losses, metrics
  Enhancement/                   LoL enhancement scripts and Options configs
实验结果_PatchCore15类/          PatchCore experiment results (CSV/JSON manifests, metrics)
manuscript/                      Paper source (.tex/.bib), audit JSONs, submission checks
WileyDesign/                     Wiley-formatted manuscript source and figures
```

## Reproduce

1. Install MIRNetv2 dependencies and download MVTec AD (and MPDD) data. Data is **not** stored
   in this repository because of size and licensing.
2. Set the dataset path in the relevant config under `MIRNetv2/industrial/configs/`.
3. Run the pipeline:

   ```bash
   cd MIRNetv2
   python industrial/run.py --config industrial/configs/mvtec_full.json
   ```

4. See `工业设备巡检低照度增强复现报告.md` for the exact reproduction protocol, hardware,
   degradation model, evaluation definitions, and the full multi-seed audit.

## Audit Note

Older anomaly-detection features were extracted without the official torchvision ImageNet
normalization, so earlier AUROC values (0.813 / 0.890) are invalidated and are not used as paper
conclusions. Current results use torchvision official preprocessing, per-class normal-sample fitting,
and unweighted category macro-average; all pre-registered configs passed strict validation.

## Upstream Baseline

- Official code: `swz30/MIRNetv2`
- Baseline commit referenced in the reproduction report: `9014049f051ad714eed8b816f23a571e986a46c7`
- Paper DOI: `10.1109/TPAMI.2022.3167175`

## License

Model code follows the upstream MIRNetv2 license (see `MIRNetv2/LICENSE.md`). Reports and
experimental artifacts are provided for research reproducibility.
