# PatchCore MVTec AD 15-Class Training-Range Report

## Protocol

- Fifteen independent category-local PatchCore detectors were fitted on clear `train/good` only.
- Each saved category detector was reused unchanged for dark, MIRNet-v2, and ICRA test domains.
- Test identity and label order came from the preflighted canonical manifests; no test adaptation was used.

## Macro image-level AUROC

- Dark: 0.502957
- MIRNet-v2: 0.489249 (delta vs dark -0.013708)
- ICRA: 0.535887 (delta vs dark +0.032930)

## Audit and runtime

- Categories: 15; domains: 3; train/test overlap: 0.
- Labels used for fitting: false; restored test outputs used for fitting: false; test adaptation: false.
- Runtime: 854.028 s; evaluation: 158.427 s.
- Peak CUDA memory allocated (not reserved): 187.556 MiB.
- Persisted validator: PASS.

This report is limited to the fixed MVTec-15 training-range matrix.
