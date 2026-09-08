# Low-Light Industrial Enhancement & Anomaly Detection

Reproducible code for low-light industrial enhancement and anomaly detection. It vendors the
MIRNet-v2 codebase plus an illumination-conditioned residual adapter (ICRA-MIRNet) under
`MIRNetv2/`, and the resulting experiment outputs under `实验结果_PatchCore15类/`.

## Repository Layout

```text
MIRNetv2/                        Pinned source (upstream swz30/MIRNetv2) + industrial additions
  industrial/                    ICRA-MIRNet adapter, runner, and experiment configs
  basicsr/                       Model architecture, losses, metrics
  Enhancement/                   LoL enhancement scripts and Options configs
实验结果_PatchCore15类/          Experiment results (CSV/JSON manifests, metrics)
```

## Reproduce

1. Install the MIRNetv2 dependencies and download the MVTec AD (and MPDD) data. Data and model
   weights are **not** stored here because of size and licensing.
2. Set the dataset path in the relevant config under `MIRNetv2/industrial/configs/`.
3. Run the pipeline:

   ```bash
   cd MIRNetv2
   python industrial/run.py --config industrial/configs/mvtec_full.json
   ```

Exact experiment parameters (seeds, micro-batch count, degradation settings, evaluation protocol)
are in the config JSON files under `MIRNetv2/industrial/configs/` and the results under
`实验结果_PatchCore15类/`.

## License

Model code follows the upstream MIRNetv2 license (see `MIRNetv2/LICENSE.md`); the repository is
MIT-licensed (see `LICENSE`).
