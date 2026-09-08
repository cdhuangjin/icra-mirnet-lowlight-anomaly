# 实验交接（详细版）：Phase 4 / Phase 4.1 — Restore-or-Bypass Gate 一致性审计

> **交接日期**：2026-09-07
> **论文主线**：*When Does Low-Light Restoration Help Industrial Anomaly Detection?*
> **项目根**：`C:\Users\PC\Documents\Codex\Reproduction003`
> **实验工作树**：`C:\Users\PC\Documents\Codex\Reproduction003\MIRNetv2\.worktrees\industrial-lowlight`
> **交接范围**：Phase 3（冻结主实验）+ Phase 4（label-free Restore-or-Bypass Gate，失败）+ Phase 4.1（一致性审计、最小修复、重跑）

---

## 0. 一句结论（先读这个）

当前实验（Phase 4 / Phase 4.1）**已收尾，结论为科研上的真实负结果**：

- **PHASE4_FAIL**：只在 normal 数据上监督的 label-free Restore-or-Bypass Gate，修复后仍是选择性路由（59–80% 样本走 restore，不是坍缩），但 2/3 个 severity 下打不过 Always-ICRA（均值 Gate − Always = **−0.0276**）。
- 原始 Phase 4 报出的 `restore_rate = 0` 是**假象**：是 `aggregate()` 里 `"1" == 1` 恒为 `False` 造成的**聚合计数 bug**，不是路由坍缩、也不是退化不匹配。
- 科研归因：**PHASE4_1_SURROGATE_MISMATCH** —— normal 图像上的 detector-consistency utility 代理，不能可靠迁移到异常区分 utility。
- 工程一致性审计全部通过：**PHASE4_1_VALIDATOR_PASS（17/17 check）**。
- **不再建议 Phase 4.2**。项目已进入正式 SCI 论文写作阶段（`WileyDesign` 排版）。

---

## 1. 研究背景与协议

### 1.1 研究问题
低照度增强对工业异常检测**是否总是有利**？能否用 label-free 策略自动决定“该增强（restore）还是该跳过（bypass）”，在不使用异常标签、不做 test-time 自适应的情况下超过“无条件增强”？

结论链：

1. **Phase 3（已冻结）**：低照度增强并非普遍有益；下游收益取决于退化强度、增强方法、类别。
2. **Phase 4**：只凭 normal train/good 构造 utility pseudo-label，训练 Restore-or-Bypass Gate。
3. **Phase 4.1**：审计 Phase 4 的“`restore_rate=0` 但 Gate≠Dark”矛盾 → 定位聚合 bug → 最小修复 → 重跑 → 确认负结果。

### 1.2 数据与评测协议

| 项 | 值 |
|---|---|
| 数据集 | MVTec AD，15 类：`bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper` |
| 检测器 | 官方 PatchCore：ResNet-50 Wide（WR50）layer2+layer3，patch size 3，resize 256 → crop 224；coreset 10%；batch_size 1 |
| 增强模型 | ICRA-MIRNet（工业低照度适配版），对照为 MIRNet-v2（LOL 预训练） |
| 全局种子 | `20260830`；统计 bootstrap = 10000，bootstrap 种子 = `20260830` |
| 基线 | Dark（不增强）、Always-ICRA（无条件增强）、Gate-ICRA（Gate 选择） |
| 统计单位 | category（15 类）为 paired unit；两尾 Wilcoxon signed-rank，`zero_method=wilcox`；rank-biserial 排除零差、平均 tie |
| 窗口 | Windows 11，Python 3.12.10，torch 2.13.0+cu130，numpy 2.4.4，opencv 5.0.0，CUDA 13.0，GPU = NVIDIA GeForce RTX 5060 |

### 1.3 低照度退化定义（冻结）

| Severity | Gamma | Read noise | 备注 |
|---|---|---|---|
| `mild` | `[1.2, 2.0]` | `[0.0, 0.01]` | 最接近原图 |
| `training_range` | `[2.0, 4.0]` | `[0.0, 0.02]` | 训练/测试覆盖范围 |
| `severe` | `[4.0, 5.0]` | `[0.02, 0.05]` | 最暗、噪声最大 |

Phase 3 与 Phase 4 使用**同一条冻结 dark cache**（`runs/reliability_v2/enhancement_outputs/dark/{severity}/`），image-SHA256 逐样本一致（见 §8.7）。退化 generator 与 ICRA checkpoint 哈希记录在 `patchcore_official/protocol.json`。

### 1.4 无泄漏声明

- 每个类别 detector 仅在 clear train/good 上拟合一次；mild/severe 复用序列化 detector（zero fit call）。
- 拟合只接收 clear train/good 路径；labels 与 restored test 输出不参与拟合；`test_adaptation=False`。
- train/test 路径与内容重叠：**0**。
- Gate 推断只读 dark 图像；`predict()` 无 label/severity/category 参数；threshold 仅在 gate-val 上冻结。

---

## 2. 恢复模型对比（ICRA 变体，论文支撑）

> 来源：`runs/final-paper-results/final_results.json`（ICRA-MIRNet 三随机种子均值 ± std；其余为单 seed）。

| 方法 | PSNR | SSIM | AUROC_Dark | AUROC_Restored | 延迟(秒) | 峰值显存(MiB) |
|---|---:|---:|---:|---:|---:|---:|
| LOL 预训练 MIRNet-v2 | 18.168 | 0.616 | 0.667 | 0.704 | 3.775 | 350.5 |
| Fixed-gate 微调 | 28.581 | 0.896 | 0.667 | 0.747 | 4.062 | 386.2 |
| ICRA-MIRNet 无 L∇ | 28.744 | 0.891 | 0.667 | 0.750 | 3.850 | 386.2 |
| **ICRA-MIRNet** | **28.472 ± 0.243** | **0.895 ± 0.002** | **0.671 ± 0.005** | **0.738 ± 0.007** | **3.034 ± 0.886** | **386.2 ± 0.0** |

> 说明：这是低照度增强方法本身的图像质量 + 下游 AUROC 对比，用于说明“为什么选 ICRA 作为增强器”。它和 Phase 3 的 PatchCore × severity 实验是两个维度（前者是增强器横向对比，后者是 severity 纵向对比）。`mild` 与 `severe` 两个额外 run 见 `final_results.json`（mild: restored 0.712 / severe: restored 0.658）。

---

## 3. Phase 3 全量结果（已冻结）

> 权威来源：`runs/reliability_v2/patchcore_official/PATCHCORE_MVTEC15_SEVERITY_REPORT.md`、`patchcore_severity_results.json`、`patchcore_severity_by_category.csv`。

### 3.1 三 severity 宏平均（论文主表）

| Severity | Dark | MIRNet | Δ MIRNet | ICRA | Δ ICRA |
|---|---:|---:|---:|---:|---:|
| mild | 0.944966 | 0.928972 | −0.015994 | 0.962859 | **+0.017893** |
| training_range | 0.799974 | 0.898658 | +0.098685 | 0.937793 | **+0.137820** |
| severe | 0.756876 | 0.775847 | +0.018971 | 0.775843 | **+0.018967** |

### 3.2 提升 / 退化类别数（vs Dark）

| Severity | 方法 | 提升 | 退化 | 持平 |
|---|---|---:|---:|---:|
| mild | mirnetv2 | 7 | 8 | 0 |
| mild | icra | 12 | 3 | 0 |
| training_range | mirnetv2 | 14 | 1 | 0 |
| training_range | icra | **15** | **0** | 0 |
| severe | mirnetv2 | 9 | 6 | 0 |
| severe | icra | 9 | 6 | 0 |

### 3.3 逐类 AUROC（mild）

| Category | Dark | MIRNet | ICRA | Δ MIRNet | Δ ICRA |
|---|---:|---:|---:|---:|---:|
| bottle | 0.932540 | 0.984921 | 0.949206 | +0.052381 | +0.016667 |
| cable | 0.967766 | 0.985757 | 0.992879 | +0.017991 | +0.025112 |
| capsule | 0.921021 | 0.663742 | 0.758277 | −0.257280 | −0.162744 |
| carpet | 0.984350 | 0.974719 | 0.985955 | −0.009631 | +0.001605 |
| grid | 0.975773 | 0.965748 | 0.977444 | −0.010025 | +0.001671 |
| hazelnut | 0.997500 | 1.000000 | 1.000000 | +0.002500 | +0.002500 |
| leather | 0.999321 | 1.000000 | 1.000000 | +0.000679 | +0.000679 |
| metal_nut | 0.996579 | 0.996090 | 0.995112 | −0.000489 | −0.001466 |
| pill | 0.792144 | 0.630387 | 0.917076 | −0.161757 | +0.124932 |
| screw | 0.940357 | 0.968436 | 0.942406 | +0.028080 | +0.002050 |
| tile | 1.000000 | 0.964286 | 0.988817 | −0.035714 | −0.011183 |
| toothbrush | 0.877778 | 0.858333 | 0.980556 | −0.019444 | +0.102778 |
| transistor | 0.968750 | 0.999167 | 0.996667 | +0.030417 | +0.027917 |
| wood | 0.986842 | 0.945614 | 0.992105 | −0.041228 | +0.005263 |
| zipper | 0.833771 | 0.997374 | 0.966387 | +0.163603 | +0.132616 |

### 3.4 逐类 AUROC（training_range）

| Category | Dark | MIRNet | ICRA | Δ MIRNet | Δ ICRA |
|---|---:|---:|---:|---:|---:|
| bottle | 0.930952 | 0.986508 | 0.975397 | +0.055556 | +0.044444 |
| cable | 0.809033 | 0.917916 | 0.919228 | +0.108883 | +0.110195 |
| capsule | 0.623454 | 0.549262 | 0.641404 | −0.074192 | +0.017950 |
| carpet | 0.922953 | 0.958266 | 0.988764 | +0.035313 | +0.065811 |
| grid | 0.911445 | 0.948204 | 0.977444 | +0.036759 | +0.065998 |
| hazelnut | 0.950000 | 0.998571 | 1.000000 | +0.048571 | +0.050000 |
| leather | 0.925611 | 0.994905 | 0.984715 | +0.069293 | +0.059103 |
| metal_nut | 0.789834 | 0.928641 | 0.949169 | +0.138807 | +0.159335 |
| pill | 0.550464 | 0.653028 | 0.903710 | +0.102564 | +0.353246 |
| screw | 0.608116 | 0.857143 | 0.879484 | +0.249026 | +0.271367 |
| tile | 0.992785 | 0.995671 | 0.997475 | +0.002886 | +0.004690 |
| toothbrush | 0.850000 | 0.930556 | 0.994444 | +0.080556 | +0.144444 |
| transistor | 0.545417 | 0.867083 | 0.902083 | +0.321667 | +0.356667 |
| wood | 0.953509 | 0.976316 | 0.990351 | +0.022807 | +0.036842 |
| zipper | 0.636029 | 0.917805 | 0.963235 | +0.281775 | +0.327206 |

### 3.5 逐类 AUROC（severe）

| Category | Dark | MIRNet | ICRA | Δ MIRNet | Δ ICRA |
|---|---:|---:|---:|---:|---:|
| bottle | 0.894444 | 0.906349 | 0.903968 | +0.011905 | +0.009524 |
| cable | 0.659295 | 0.839205 | 0.753748 | +0.179910 | +0.094453 |
| capsule | 0.635421 | 0.631432 | 0.572397 | −0.003989 | −0.063024 |
| carpet | 0.786918 | 0.795345 | 0.965490 | +0.008427 | +0.178571 |
| grid | 0.507101 | 0.538012 | 0.579783 | +0.030911 | +0.072682 |
| hazelnut | 0.913571 | 0.937500 | 0.969286 | +0.023929 | +0.055714 |
| leather | 0.909307 | 0.896399 | 0.907609 | −0.012908 | −0.001698 |
| metal_nut | 0.677908 | 0.650049 | 0.667155 | −0.027859 | −0.010753 |
| pill | 0.901800 | 0.812602 | 0.747136 | −0.089198 | −0.154664 |
| screw | 0.430826 | 0.561590 | 0.543349 | +0.130765 | +0.112523 |
| tile | 0.994228 | 0.993506 | 0.993867 | −0.000722 | −0.000361 |
| toothbrush | 0.775000 | 0.836111 | 0.797222 | +0.061111 | +0.022222 |
| transistor | 0.742083 | 0.672500 | 0.622917 | −0.069583 | −0.119167 |
| wood | 0.930702 | 0.942105 | 0.961404 | +0.011404 | +0.030702 |
| zipper | 0.594538 | 0.625000 | 0.652311 | +0.030462 | +0.057773 |

### 3.6 Phase 3 统计检验（category 为 unit，bootstrap 10000，seed 20260830）

| Severity | 方法 | Mean Δ | Median Δ | 95% CI | Wilcoxon p | Rank-biserial |
|---|---|---:|---:|---|---:|---:|
| mild | mirnetv2 | −0.015994 | −0.000489 | [−0.063920, +0.027128] | 0.761536 | −0.100 |
| mild | icra | +0.017893 | +0.002500 | [−0.017666, +0.050970] | 0.047913 | +0.583 |
| training_range | mirnetv2 | +0.098685 | +0.069293 | [+0.048175, +0.154789] | 0.001526 | +0.867 |
| training_range | icra | +0.137820 | +0.065998 | [+0.078814, +0.201464] | **6.1e-05** | **+1.000** |
| severe | mirnetv2 | +0.018971 | +0.011404 | [−0.012259, +0.053395] | 0.330261 | +0.300 |
| severe | icra | +0.018967 | +0.022222 | [−0.023067, +0.059413] | 0.359131 | +0.283 |

### 3.7 Phase 3 研究者解读

- ICRA 在 `training_range` 上 15/15 类别提升，均值 +0.1378（p≈6e-5，rank-biserial 1.0）——条件性最明显的证据。
- mild 与 severe 下收益显著缩小（severe 仅 +0.019，无统计显著），支持“增强并非普遍有益”。
- 跨 severity 的 Δ 范围：`mirnetv2=0.1147`，`icra=0.1199`，存在方法在跨 severity 出现均值方向反转 → 描述性支持 severity-dependence。
- 仅为单 seed、15 个 paired unit、合成低照度，无 pixel-level / 多检测器泛化主张。

### 3.8 PatchCore detector 指纹（15 类）

来自 `patchcore_official`（官方实现，clear train/good 拟合一次，mild/severe 复用）：

| Category | Detector Fingerprint |
|---|---|
| bottle | `ef6a73606a3133a403f17f8780906ef750b43f44d26672ebe859358beffa5133` |
| cable | `af5a1091e602f49c901e97bc5c4cb27e1f11794bc3ceb0bb766c184d51aeeec3` |
| capsule | `77405b6f9ff1e4787c238cbd40574d53797a3c49f75f90ecb9b2215af7f2e298` |
| carpet | `005cd7190d2e395983c3e7ae94f8464ebf04e2d0832c0adb56143eaede2318b9` |
| grid | `971a7eae7f78fe69e9bb3b558ba2c0aaadac5bad80d9ba19b7096f5dc51cf2d4` |
| hazelnut | `8b0f1790c3cce3a3890ded6d1c356dfa05b292443d5f869783436f9367b93f6b` |
| leather | `dde173d3f6c14729b95b4123aeee9f4ffa012abc5cb717ed50b4b7c2bc43bf79` |
| metal_nut | `dd100ab2b7e44624613438cf3724233defa849ae2d176dfd06ea025097a41160` |
| pill | `ced15c517b155d30de144874f2d1f4b7cfd4d29f581d4c43392f93f74d537fab` |
| screw | `5c5b1592073a8a9d471bbdb439c110cb74eb0a34c320bd5b4db57045685009f4` |
| tile | `85f0c81f9cadda4d61572afb9ecdf6e2774616147e274efba2e61222b63c8aef` |
| toothbrush | `503f2d59c9fcd27f4e5ec6b629eed406ea27adb7f155d3b05bd860f9f1d7c6cf` |
| transistor | `fea062013a7800795d2f42408e6da0ebc06d72be00bab5dbfb3b2225c57696b0` |
| wood | `40d0d16228cadbe2cb5072869b0e8f5aa52d7e5d02103df19a1d6252687ea7b8` |
| zipper | `f2387b090a801afab1e62b36e6760436d205a67e78f09356fb3dd401f6d814b4` |

---

## 4. Phase 4 设计（Gate）

### 4.1 目标
> 只读 dark 图像、无异常标签、无 test-time 自适应的 label-free 决策：`restore`（喂 ICRA）或 `bypass`（直接用 dark）。

### 4.2 Gate 配置（冻结）

| 项 | 值 |
|---|---|
| 分类器 | `LogisticRegression(C=1.0, class_weight="balanced", solver=lbfgs, max_iter=2000)` |
| 预处理 | `StandardScaler`（23 维） |
| threshold | `0.55`（在 gate-val 上按 balanced accuracy 选择；tie 取最接近 0.5，再低） |
| 决策 | `decision = 1 if p_restore >= 0.55 else 0` |
| 划分 | gate_train n=17400 / gate_val n=4374（按原始 clear 图像 group-disjoint） |
| 其它 | `delta=0.0`，`replicates_per_severity=2`，`validation_fraction=0.2` |
| threshold 候选 | `[0.05,0.10,…,0.95]`（19 档） |
| collapse bounds | `[0.01, 0.99]` |

### 4.3 Gate 特征（23 维 image-only）

`mean_luminance, median_luminance, p05, p10, p25, p75, p90, p95, dark05, dark10, dark20, std_luminance, p90_minus_p10, rms_contrast, block_std, block_range, darkest_block, brightest_block, sobel_mean, sobel_median, high_gradient_ratio, residual_mad, residual_std`

### 4.4 Utility pseudo-label 定义

`U = |s_dark − s_clear| − |s_restored − s_clear|`；`label = 1 if U > 0 else 0`（1 = restore，0 = bypass），只用 clear train/good。方向已验证正确。

### 4.5 无泄漏

`decisions/decisions_manifest.json`：`n_decisions=5175`，`test_labels_accessed=false`。

---

## 5. Phase 4 原始上报结果（含 bug）

> 来源：`runs/reliability_v2/phase4_restore_bypass/`（已 SHA256 冻结，**未改动**）。

### 5.1 原始主表

| Severity | Dark | Always-ICRA | Gate-ICRA | overall_restore_rate |
|---|---:|---:|---:|---:|
| mild | 0.944966 | 0.962859 | 0.963588 | **0.0** |
| training_range | 0.799974 | 0.937793 | 0.880869 | **0.0** |
| severe | 0.756876 | 0.775843 | 0.749202 | **0.0** |

### 5.2 原始 summary / aggregate（buggy 值）

`results/phase4_gate_summary.csv`、`results/phase4_aggregate.json` 记录 `overall_restore_rate = 0.0`、`mean_restore_rate=[0,0,0]`，但 `Gate-ICRA ≠ Dark`（如 mild Bottle 0.942063 vs Dark 0.932540；training_range screw 0.855913 vs Dark 0.608116；severe tile 0.875541 vs Dark 0.994228）。

### 5.3 原始 Phase 4 checker 状态

`phase4_validator.json`（旧校验器）：`PASS`（gate_api/decisions_no_label/required_artifacts/counts/auc_recompute/reproducibility 全 ok）。**这说明旧校验器没有覆盖 restore-rate 的 bug**，因此在 checker 层面检测不到。

---

## 6. 关键逻辑矛盾（Phase 4.1 的起点）

若 `n_restore == 0`，则每个样本都应走 dark 分支，Gate-ICRA 分数向量必等于 Dark 分数向量、AUROC 必相等。但原始报告 `restore_rate=0` 的同时 Gate-ICRA ≠ Dark，这在逻辑上矛盾。

Phase 4.1 的任务：查清矛盾出自哪一层（feature / scaler / model / threshold / decision→分支 / score 来源 / sample 对齐 / 聚合 / degradation realization / AUROC）。

---

## 7. Phase 4.1 审计方法（步骤 + 产物）

严格按 `AUDIT FIRST → IDENTIFY ROOT CAUSE → MINIMAL FIX → RE-RUN SAME PHASE 4 → STOP`。所有改动落到 `runs/reliability_v2/phase4_1_gate_audit/`，原始 Phase 4 不动。

1. **冻结原始 Phase 4**：对 27 个原件做 `path/size/sha256` 快照 → `original_snapshot/manifest.json`。
2. **Utility label 分布审计** → `audit/utility_label_summary.json`、`audit/utility_label_distribution.csv`。
3. **Feature schema 审计** → `audit/canonical_feature_schema.json`。
4. **StandardScaler 审计** → `audit/scaler_audit.json`。
5. **Gate model 重算** → `audit/gate_model_recompute.json`。
6. **Threshold 审计** → `audit/threshold_audit.json`。
7. **逐样本 Trace** → `sample_traces/{mild,training_range,severe}_trace.csv`。
8. **决策→分数不变量检查** → `tests/test_gate_bypass_score_identity.py`、`test_gate_restore_score_identity.py`、`test_zero_restore_equals_dark.py`、`test_gate_macro_identity_under_zero_restore.py`。
9. **Phase 3 vs Phase 4 身份检查** → `audit/phase3_vs_phase4_dark_identity.csv`、`audit/phase3_vs_phase4_icra_identity.csv`。
10. **根因复现** → `audit/restore_count_bug_reproduction.json`。
11. **特征 / 概率分布偏移** → `distributions/*`。
12. **Logistic 分解** → `audit/logit_contribution_low20.csv`、`audit/logit_contribution_high20.csv`。
13. **最小修复重跑** → `repaired_results/*`。
14. **17 项校验** → `phase4_1_validator.json`。
15. **主报告** → `PHASE4_1_GATE_AUDIT_REPORT.md`。

---

## 8. Phase 4.1 审计发现

### 8.1 Utility label 分布

| 项 | 值 |
|---|---:|
| 总记录数 | 21774（train 17400 / val 4374） |
| 总 restore 阳性率 | 0.8845 |
| train 阳性率 | 0.8843 |
| val 阳性率 | 0.8855 |
| mild 阳性率 | 0.7629 |
| training_range 阳性率 | 0.9956 |
| severe 阳性率 | 0.8950 |

按类别阳性率（`audit/utility_label_distribution.csv`）：

| Category | Overall | mild | training_range | severe |
|---|---:|---:|---:|---:|
| bottle | 0.9960 | 1.000 | 1.000 | 0.9880 |
| cable | 0.9799 | 0.9397 | 1.000 | 1.000 |
| capsule | 0.7336 | 0.3402 | 0.9840 | 0.8767 |
| carpet | 0.9833 | 0.9500 | 1.000 | 1.000 |
| grid | 0.9331 | 0.8750 | 1.000 | 0.9242 |
| hazelnut | 0.8662 | 0.8286 | 1.000 | 0.7698 |
| leather | 0.9714 | 0.9143 | 1.000 | 1.000 |
| metal_nut | 0.8091 | 0.4636 | 0.9818 | 0.9818 |
| pill | 0.6941 | 0.5562 | 0.9850 | 0.5412 |
| screw | 0.8073 | 0.4359 | 0.9922 | 0.9938 |
| tile | 0.9428 | 0.8283 | 1.000 | 1.000 |
| toothbrush | 0.8861 | 0.7167 | 0.9667 | 0.9750 |
| transistor | 0.7707 | 0.8357 | 1.000 | 0.4765 |
| wood | 0.9204 | 0.7611 | 1.000 | 1.000 |
| zipper | 1.0000 | 1.000 | 1.000 | 1.000 |

### 8.2 Feature schema 审计

- 特征数 = **23**；`utility_labels` 的 `f0..f22` 与 canonical 顺序一致：**True**。
- trace 的 `raw_feature_XX` / `std_feature_XX` 列与 canonical 顺序一致。
- `canonical_feature_schema.json`：`expected_23=true`，`utility_labels_matches_canonical=true`。

### 8.3 StandardScaler 审计

- `n_features_in_ = 23`；frozen `feature_scaler.pkl` 与模型 pipeline scaler 的 `mean` / `scale` 相等：**True**。
- 无 test-time fit、无 re-fit、无重复/漏 transform。
- 正确路径：`X_std = scaler.transform(X)`；`p = model.predict_proba(X_std)`。

### 8.4 Threshold 审计

- 存储 threshold = **0.55**（`is_0_55=true`）；`Gate.predict` 用 `>=`（`True`）。
- 无 `>`/`>=` 混用、无 `round`/`argmax`、无 per-category / per-severity threshold。

### 8.5 Gate model 重算（与冻结值完全一致）

| Metric | 存储值 | 重算值 | 一致 |
|---|---:|---:|:--:|
| balanced_accuracy | 0.74421335 | 0.74421335 | ✔ |
| roc_auc | 0.82912770 | 0.82912770 | ✔ |
| pr_auc | 0.97219198 | 0.97219198 | ✔ |
| precision | 0.96407837 | 0.96407837 | ✔ |
| recall | 0.68603150 | 0.68603150 | ✔ |
| f1 | 0.80162921 | 0.80162921 | ✔ |
| accuracy | 0.69935985 | 0.69935985 | ✔ |

`matches_stored = True`。训练阶段无 model/scaler bug。

### 8.6 Sample 对齐审计

- 决策与 Phase 3 分数按 `(domain, canonical_id)` key join，非 list 位置。
- 每个 severity 的 `sample_id` 唯一（`sample_ids_unique=ok`）；score join 为 key-based（`score_join_keyed=ok`）。

### 8.7 Phase 3 vs Phase 4 身份一致性（逐样本）

| 比较 | 样本数 | image-SHA256 不匹配 | 结论 |
|---|---:|---:|---|
| Phase 3 dark vs Phase 4 gate dark | 5175 | **0** | 同一冻结 cache |
| Phase 3 ICRA vs Phase 4 gate ICRA | 5175 | **0** | 同一冻结 cache |

→ **无 degradation realization mismatch，无 sample-order/join bug。**

### 8.8 根因复现（bug reproduction）

`audit/restore_count_bug_reproduction.json`：

| Severity | n_samples | 上报(错误)restore | 实际 positive 决策 | buggy rate | 正确 rate |
|---|---:|---:|---:|---:|---:|
| mild | 1725 | 0 | 1020 | 0.000 | 0.5913 |
| training_range | 1725 | 0 | 1388 | 0.000 | 0.8046 |
| severe | 1725 | 0 | 886 | 0.000 | 0.5136 |

---

## 9. 根因（Root Cause）

**聚合计数 bug。** 在 `reliability_v2/gate/evaluate_gate.py::aggregate()`：

```python
restore = sum(1 for d in by_id.values() if d['gate_decision'] == 1)   # 错
```

`gate_decision` 经 `csv.DictReader` 读入为**字符串** `"1"`，`"1" == 1` 恒为 `False`，于是每个 cell 的 `n_restore=0`、`restore_rate=0`。但 `final_anomaly_score` 用的是真实（正确）的 dark/ICRA 混合分支分数，故 Gate-ICRA AUROC 确实 ≠ Dark。**“矛盾”是 restore-rate 上报 bug 造成的人为假象；真正 pipeline 无矛盾。**

其余环节（feature、scaler、threshold、Phase3/Phase4 image identity、key-based join、detector 指纹、标签泄漏）全部通过。

---

## 10. 最小修复

仅改一行：

```python
restore = sum(1 for d in by_id.values() if int(d['gate_decision']) == 1)  # 修
```

未改 classifier / feature / threshold / pseudo-label / degradation / PatchCore / ICRA。冻结的 gate model、scaler、threshold 原样复用（**MODEL_REUSED**），因训练阶段已验证正确。原始 Phase 4 产物未覆盖；修复输出隔离到 `phase4_1_gate_audit/repaired_results/`。

---

## 11. 修复后主结果（Phase 4.1 重跑）

> `repaired_results/phase4_gate_summary.csv`、`phase4_gate_results.json`、`phase4_gate_by_category.csv`

| Severity | Dark | Always-ICRA | Gate-ICRA | Gate−Dark | Gate−Always | Restore%（macro） | Restore%（pooled） |
|---|---:|---:|---:|---:|---:|---:|---:|
| mild | 0.944966 | 0.962859 | 0.963588 | +0.018622 | +0.000729 | 63.16% | 59.13% |
| training_range | 0.799974 | 0.937793 | 0.880869 | +0.080895 | −0.056925 | 82.35% | 80.46% |
| severe | 0.756876 | 0.775843 | 0.749202 | −0.007675 | −0.026641 | 51.59% | 51.36% |

判断：

- **非坍缩**：restore 59–80%（选择性），非 routing collapse、也非 always-restore collapse。
- **打不过 Always**：training_range（−0.0569）、severe（−0.0266）均差；仅 mild 微胜（+0.0007）。
- `mild/pill` 是唯一 `n_restore=0` 的 cell（Dark=Gate=0.792144，满足 zero-restore invariant）。

### 11.1 修复后逐类（含 restore 计数）

列：`category, dark, always, gate, gate−always, restore_rate, n_restore, n_bypass, n_test`

**mild**

| Category | Dark | Always | Gate | Gate−Always | Restore% | n_res | n_byp | n_test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 0.932540 | 0.949206 | 0.942063 | −0.007143 | 93.98 | 78 | 5 | 83 |
| cable | 0.967766 | 0.992879 | 0.982946 | −0.009933 | 82.00 | 123 | 27 | 150 |
| capsule | 0.921021 | 0.758277 | 0.946550 | +0.188273 | 4.55 | 6 | 126 | 132 |
| carpet | 0.984350 | 0.985955 | 0.982343 | −0.003612 | 96.58 | 113 | 4 | 117 |
| grid | 0.975773 | 0.977444 | 0.971596 | −0.005848 | 64.10 | 50 | 28 | 78 |
| hazelnut | 0.997500 | 1.000000 | 1.000000 | +0.000000 | 100.00 | 110 | 0 | 110 |
| leather | 0.999321 | 1.000000 | 1.000000 | +0.000000 | 70.16 | 87 | 37 | 124 |
| metal_nut | 0.996579 | 0.995112 | 0.997556 | +0.002444 | 51.30 | 59 | 56 | 115 |
| pill | 0.792144 | 0.917076 | 0.792144 | −0.124932 | 0.00 | 0 | 167 | 167 |
| screw | 0.940357 | 0.942406 | 0.926829 | −0.015577 | 8.13 | 13 | 147 | 160 |
| tile | 1.000000 | 0.988817 | 0.984848 | −0.003968 | 70.09 | 82 | 35 | 117 |
| toothbrush | 0.877778 | 0.980556 | 0.980556 | +0.000000 | 97.62 | 41 | 1 | 42 |
| transistor | 0.968750 | 0.996667 | 0.996667 | +0.000000 | 100.00 | 100 | 0 | 100 |
| wood | 0.986842 | 0.992105 | 0.983333 | −0.008772 | 8.86 | 7 | 72 | 79 |
| zipper | 0.833771 | 0.966387 | 0.966387 | +0.000000 | 100.00 | 151 | 0 | 151 |

**training_range**

| Category | Dark | Always | Gate | Gate−Always | Restore% | n_res | n_byp | n_test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 0.930952 | 0.975397 | 0.933333 | −0.042063 | 74.70 | 62 | 21 | 83 |
| cable | 0.809033 | 0.919228 | 0.893553 | −0.025675 | 91.33 | 137 | 13 | 150 |
| capsule | 0.623454 | 0.641404 | 0.580774 | −0.060630 | 54.55 | 72 | 60 | 132 |
| carpet | 0.922953 | 0.988764 | 0.879615 | −0.109149 | 65.81 | 77 | 40 | 117 |
| grid | 0.911445 | 0.977444 | 0.964077 | −0.013367 | 97.44 | 76 | 2 | 78 |
| hazelnut | 0.950000 | 1.000000 | 1.000000 | +0.000000 | 98.18 | 108 | 2 | 110 |
| leather | 0.925611 | 0.984715 | 0.984715 | +0.000000 | 100.00 | 124 | 0 | 124 |
| metal_nut | 0.789834 | 0.949169 | 0.773216 | −0.175953 | 69.57 | 80 | 35 | 115 |
| pill | 0.550464 | 0.903710 | 0.516912 | −0.386798 | 26.95 | 45 | 122 | 167 |
| screw | 0.608116 | 0.879484 | 0.855913 | −0.023570 | 90.63 | 145 | 15 | 160 |
| tile | 0.992785 | 0.997475 | 0.997475 | +0.000000 | 94.87 | 111 | 6 | 117 |
| toothbrush | 0.850000 | 0.994444 | 0.994444 | +0.000000 | 95.24 | 40 | 2 | 42 |
| transistor | 0.545417 | 0.902083 | 0.902083 | +0.000000 | 100.00 | 100 | 0 | 100 |
| wood | 0.953509 | 0.990351 | 0.973684 | −0.016667 | 75.95 | 60 | 19 | 79 |
| zipper | 0.636029 | 0.963235 | 0.963235 | +0.000000 | 100.00 | 151 | 0 | 151 |

**severe**

| Category | Dark | Always | Gate | Gate−Always | Restore% | n_res | n_byp | n_test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 0.894444 | 0.903968 | 0.869048 | −0.034921 | 59.04 | 49 | 34 | 83 |
| cable | 0.659295 | 0.753748 | 0.714393 | −0.039355 | 87.33 | 131 | 19 | 150 |
| capsule | 0.635421 | 0.572397 | 0.661747 | +0.089350 | 17.42 | 23 | 109 | 132 |
| carpet | 0.786918 | 0.965490 | 0.767255 | −0.198234 | 1.71 | 2 | 115 | 117 |
| grid | 0.507101 | 0.579783 | 0.534670 | −0.045113 | 35.90 | 28 | 50 | 78 |
| hazelnut | 0.913571 | 0.969286 | 0.959286 | −0.010000 | 79.09 | 87 | 23 | 110 |
| leather | 0.909307 | 0.907609 | 0.874660 | −0.032948 | 61.29 | 76 | 48 | 124 |
| metal_nut | 0.677908 | 0.667155 | 0.582600 | −0.084555 | 50.43 | 58 | 57 | 115 |
| pill | 0.901800 | 0.747136 | 0.802510 | +0.055374 | 26.95 | 45 | 122 | 167 |
| screw | 0.430826 | 0.543349 | 0.525313 | −0.018036 | 51.25 | 82 | 78 | 160 |
| tile | 0.994228 | 0.993867 | 0.875541 | −0.118326 | 32.48 | 38 | 79 | 117 |
| toothbrush | 0.775000 | 0.797222 | 0.788889 | −0.008333 | 80.95 | 34 | 8 | 42 |
| transistor | 0.742083 | 0.622917 | 0.692083 | +0.069167 | 52.00 | 52 | 48 | 100 |
| wood | 0.930702 | 0.961404 | 0.937719 | −0.023684 | 37.97 | 30 | 49 | 79 |
| zipper | 0.594538 | 0.652311 | 0.652311 | +0.000000 | 100.00 | 151 | 0 | 151 |

---

## 12. 分布偏移（诊断）

### 12.1 Gate 概率分布（五群体）

| Population | n | mean | std | min | p05 | p50 | p95 | max | frac≥0.5 | frac≥0.55 | frac≥0.6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gate_train | 17400 | 0.6194 | 0.2411 | 4.8e-5 | 0.1586 | 0.6578 | 0.9613 | 0.9968 | 0.7036 | 0.6414 | 0.5789 |
| gate_val | 4374 | 0.6158 | 0.2383 | 4.4e-4 | 0.1657 | 0.6479 | 0.9612 | 0.9944 | 0.6948 | 0.6301 | 0.5640 |
| test_mild | 1725 | 0.5820 | 0.3028 | 1.4e-7 | 0.0472 | 0.6606 | 0.9772 | 0.9955 | 0.6162 | 0.5913 | 0.5536 |
| test_training_range | 1725 | 0.7128 | 0.1884 | 6.6e-7 | 0.3253 | 0.7525 | 0.9596 | 0.9929 | 0.8568 | 0.8046 | 0.7571 |
| test_severe | 1725 | 0.5680 | 0.1950 | 9.9e-7 | 0.2654 | 0.5575 | 0.9419 | 0.9902 | 0.6174 | 0.5136 | 0.4145 |

→ test 分布与 normal 分布重叠；training_range 明显偏 restore，severe 偏中等。**中等漂移，非路由坍缩。**

### 12.2 标准化特征均值（train 参考，z 单位）

| Feature | train | val | test_mild | test_training_range | test_severe |
|---|---:|---:|---:|---:|---:|
| mean_luminance | 0.000 | −0.011 | +0.621 | −0.127 | −0.409 |
| median_luminance | 0.000 | −0.010 | +0.529 | −0.145 | −0.368 |
| p05 | 0.000 | −0.011 | +0.614 | −0.303 | −0.455 |
| p10 | 0.000 | −0.010 | +0.601 | −0.250 | −0.441 |
| std_luminance | 0.000 | −0.006 | +0.329 | +0.063 | −0.130 |
| sobel_median | 0.000 | −0.005 | +0.323 | −0.487 | +0.054 |
| sobel_mean | 0.000 | −0.003 | +0.392 | −0.406 | +0.008 |
| block_std | 0.000 | −0.004 | +0.304 | +0.079 | −0.110 |
| block_range | 0.000 | −0.001 | +0.322 | +0.062 | −0.141 |
| high_gradient_ratio | 0.000 | +0.040 | +0.215 | +0.087 | +0.078 |
| residual_mad | 0.000 | −0.003 | +0.239 | −0.522 | +0.248 |
| residual_std | 0.000 | −0.003 | +0.178 | −0.419 | +0.385 |
| rms_contrast | 0.000 | −0.004 | −0.462 | +0.336 | +0.451 |
| darkest_block | 0.000 | −0.014 | +0.616 | −0.284 | −0.482 |
| brightest_block | 0.000 | −0.005 | +0.497 | −0.035 | −0.287 |
| dark05 / dark10 / dark20 | 0.000 | +0.00~+0.01 | −0.77~−0.53 | +0.18~+0.09 | +0.54~+0.36 |

→ 特征确实随 severity 漂移（mild 更亮、高对比；severe 更暗、对比更低），但**未压垮分类器**（概率仍落在正常区间）。

### 12.3 Logistic 分解（低/高概率样本的主导因子）

对测试集 p_restore 最低 20 与最高 20 样本分解 `logit = intercept + Σ coef_j · std_feature_j`。最低概率样本的负向主导因子：`sobel_median`、`sobel_mean`、`mean_luminance`。详见 `audit/logit_contribution_low20.csv`、`logit_contribution_high20.csv`。

---

## 13. Gate Logistic 系数（可解释性）

来源：`phase4_restore_bypass/gate_training/feature_importance.csv`（`mean/std` 为 training 特征分布；`coefficient` 为标准化特征上的 coef；`abs_rank` 按 |coef| 排序）。

| Rank | Feature | train mean | train std | coef | |coef| |
|---:|---|---:|---:|---:|---:|
| 1 | std_luminance | 0.1043 | 0.0901 | +8.916 | 8.916 |
| 2 | sobel_median | 0.0103 | 0.0053 | +5.595 | 5.595 |
| 3 | mean_luminance | 0.1331 | 0.1149 | −5.277 | 5.277 |
| 4 | sobel_mean | 0.0128 | 0.0055 | −3.578 | 3.578 |
| 5 | block_std | 0.0580 | 0.0643 | −3.537 | 3.537 |
| 6 | p10 | 0.0301 | 0.0553 | +3.354 | 3.354 |
| 7 | p05 | 0.0200 | 0.0396 | −2.577 | 2.577 |
| 8 | p90_minus_p10 | 0.2552 | 0.2241 | −2.200 | 2.200 |
| 9 | median_luminance | 0.0987 | 0.1243 | +1.651 | 1.651 |
| 10 | block_range | 0.1764 | 0.1900 | +1.525 | 1.525 |
| 11 | p90 | 0.2853 | 0.2344 | −1.312 | 1.312 |
| 12 | residual_mad | 0.0094 | 0.0045 | −1.280 | 1.280 |
| 13 | residual_std | 0.0180 | 0.0065 | −1.239 | 1.239 |
| 14 | darkest_block | 0.0571 | 0.0668 | −1.134 | 1.134 |
| 15 | brightest_block | 0.2335 | 0.2059 | +1.038 | 1.038 |
| 16 | p25 | 0.0604 | 0.0932 | +0.709 | 0.709 |
| 17 | p75 | 0.1881 | 0.1832 | −0.687 | 0.687 |
| 18 | high_gradient_ratio | 0.0016 | 0.0031 | −0.496 | 0.496 |
| 19 | dark05 | 0.4924 | 0.3265 | −0.482 | 0.482 |
| 20 | p95 | 0.3201 | 0.2475 | +0.427 | 0.427 |
| 21 | dark10 | 0.6246 | 0.3247 | −0.426 | 0.426 |
| 22 | rms_contrast | 0.9666 | 0.4592 | −0.381 | 0.381 |
| 23 | dark20 | 0.7627 | 0.2782 | −0.376 | 0.376 |

---

## 14. Harm Avoidance / Regret / Statistics

### 14.1 Harm Avoidance

| Severity | Always-ICRA 伤害 cell 数 | Gate 伤害 cell 数 | Harm Reduction |
|---|---:|---:|---:|
| mild | 3 | 5 | −0.6667 |
| training_range | 0 | 4 | N/A |
| severe | 6 | 7 | −0.1667 |
| **合计** | **9** | **16** | **−0.7778** |

→ Gate 整体未降伤害、反而略增，如实报告。

### 14.2 Regret（`BestFixed=max(Dark, Always-ICRA)`，`Regret=BestFixed−Gate`）

| Severity | mean | median | max |
|---|---:|---:|---:|
| mild | 0.010964 | 0.003612 | 0.124932 |
| training_range | 0.056925 | 0.016667 | 0.386798 |
| severe | 0.049952 | 0.034921 | 0.198234 |

### 14.3 配对统计（category 为 unit；bootstrap 10000，seed 20260830）

| Severity | 比较 | Mean Δ | Median Δ | pos/neg/zero | 95% CI | Wilcoxon p | Rank-biserial |
|---|---|---:|---:|---:|---|---:|---:|
| mild | gate vs always | +0.000729 | −0.003612 | 2/8/5 | [−0.0274, +0.0345] | 0.092601 | −0.600 |
| mild | gate vs dark | +0.018622 | +0.000978 | 9/5/1 | [+0.0008, +0.0424] | 0.177114 | +0.410 |
| training_range | gate vs always | −0.056925 | −0.016667 | 0/9/6 | [−0.1147, −0.0157] | **0.007686** | **−1.000** |
| training_range | gate vs dark | +0.080895 | +0.050000 | 11/4/0 | [+0.0222, +0.1498] | 0.025574 | +0.650 |
| severe | gate vs always | −0.026641 | −0.023684 | 3/11/1 | [−0.0629, +0.0066] | 0.177114 | −0.410 |
| severe | gate vs dark | −0.007675 | +0.007018 | 8/7/0 | [−0.0390, +0.0222] | 0.803955 | −0.083 |

→ training_range 的 Gate−Always 显著为负（p=0.0077，rank-biserial −1.0），是负结果的主要证据。

---

## 15. Validator 状态

**PHASE4_1_VALIDATOR_PASS（17/17 全 ok）**：

`snapshot_exists, originals_unchanged, feature_schema_parity, scaler_parity, gate_model_recompute, threshold_exact, sample_ids_unique, score_join_keyed, bypass_score_identity, restore_score_identity, zero_restore_invariant, phase3_dark_identity, phase3_icra_identity, detector_fingerprint, label_leakage_false, metrics_reproducible, repaired_isolated`

> 旧 Phase 4 checker（`phase4_validator.json`）为 `PASS`，但**未覆盖 restore-rate 分支**，因此无法检测该 bug；Phase 4.1 validator 补上了该缺口。

---

## 16. 最终判定与科研解读

1. **PHASE4_1_IMPLEMENTATION_BUG_FIXED** —— 原 `restore_rate=0` 是真实聚合 bug，已修复。
2. **PHASE4_FAIL** —— 修复后选择性路由在 2/3 severity 打不过 Always-ICRA（均值 Gate−Always = **−0.0276**）。
3. **PHASE4_1_SURROGATE_MISMATCH** —— normal 域 detector-consistency utility 不能作为下游异常区分的充分代理；特征/概率偏移是机制之一，但非原矛盾主因。

背景：Gate 在 normal val 上 ROC-AUC=0.829、PR-AUC=0.972、balanced-acc=0.744（明显有预测力），路由在真实特征（59–80% restore）上非退化、实现正确 → 下游差距确为 surrogate mismatch。

**不建议 Phase 4.2，停止并交由人工决策。**

---

## 17. 产物清单

### 17.1 Phase 4.1 审计（`runs/reliability_v2/phase4_1_gate_audit/`）

| 文件 | 说明 |
|---|---|
| `PHASE4_1_GATE_AUDIT_REPORT.md` | 主报告（26 节） |
| `phase4_1_validator.json` | 17 项状态 |
| `audit/canonical_feature_schema.json` | 23 维 canonical order |
| `audit/utility_label_summary.json` | 标签分布汇总 |
| `audit/utility_label_distribution.csv` | 按类别标签阳性率 |
| `audit/gate_model_recompute.json` | 指标重算 |
| `audit/scaler_audit.json` | scaler 审计 |
| `audit/threshold_audit.json` | threshold=0.55 确认 |
| `audit/restore_count_bug_reproduction.json` | 根因复现（buggy vs correct） |
| `audit/phase3_vs_phase4_dark_identity.csv` | dark 身份一致性 |
| `audit/phase3_vs_phase4_icra_identity.csv` | ICRA 身份一致性 |
| `audit/logit_contribution_low20.csv` / `..._high20.csv` | logistic 分解 |
| `distributions/raw_feature_distribution.csv` | 原始特征分布 |
| `distributions/standardized_feature_distribution.csv` | 标准化特征分布 |
| `distributions/gate_probability_distribution.csv` | 概率分布（五群体） |
| `distributions/gate_probability_by_category.csv` | 按类别概率 |
| `sample_traces/mild_trace.csv` | 逐样本 trace |
| `sample_traces/training_range_trace.csv` | 逐样本 trace |
| `sample_traces/severe_trace.csv` | 逐样本 trace |
| `repaired_results/phase4_gate_results.json` | 修复后逐类结果 |
| `repaired_results/phase4_gate_by_category.csv` | 修复后逐类 |
| `repaired_results/phase4_gate_summary.csv` | 修复后主表 |
| `repaired_results/phase4_gate_statistics.csv` | 配对统计 |
| `repaired_results/phase4_harm_avoidance.csv` | harm |
| `repaired_results/phase4_regret.csv` | regret |
| `repaired_results/repair_summary.json` | 修复摘要 + verdict |
| `original_snapshot/manifest.json` | 27 个原始文件 SHA256 |

**逐样本 trace 字段**（`sample_traces/*.csv`，每 severity 1725 行）：
`sample_id, category, defect_type, filename, severity, raw_feature_00..22, std_feature_00..22, gate_logit, gate_probability, threshold, gate_decision, selected_domain, dark_image_path, icra_image_path, dark_score, icra_score, selected_score, dark_score_source, icra_score_source, selected_score_source, dark_image_sha256, icra_image_sha256, detector_hash`

### 17.2 原始 Phase 4（冻结，`phase4_restore_bypass/`）

| 文件 | 说明 |
|---|---|
| `PHASE4_RESTORE_BYPASS_REPORT.md` | 旧报告（报 restore_rate=0） |
| `phase4_validator.json` | 旧 checker（PASS，未覆盖 bug） |
| `config.json` | Gate 配置 |
| `generation_provenance.json` | 数据生成溯源 |
| `data_status.json` / `data_queue.log` / `EXECUTION_STATUS.md` | 队列/状态 |
| `decisions/{mild,training_range,severe}.csv` | 原始决策 |
| `gate_training/feature_importance.csv` | Logistic 系数 |
| `gate_training/gate_model.pkl` | 冻结模型 |
| `gate_training/feature_scaler.pkl` | 冻结 scaler |
| `gate_training/threshold.json` | 0.55 |
| `gate_training/validation_metrics.json` | 验证指标 |
| `gate_training/utility_labels.csv` | 标签数据 |
| `gate_training/{train,val}_manifest.csv` | 划分 |
| `gate_training/batches/**` | 分批生成数据 |
| `results/phase4_gate_summary.csv` 等 | 原始（buggy）上报表 |

### 17.3 Phase 3（`runs/reliability_v2/patchcore_official/`）

`PATCHCORE_MVTEC15_SEVERITY_REPORT.md`、`patchcore_severity_results.json`、`patchcore_severity_summary.csv`、`patchcore_severity_statistics.csv`、`patchcore_severity_by_category.csv`、`protocol.json`、`provenance.json`、`events.jsonl`。

补充：项目根 `实验结果_PatchCore15类/` 是 **training_range** 单独一个少样本/中间包（不同 detector 指纹、macro 更低），**不是** Phase 3 官方结果，勿混用。

### 17.4 增强器对比（`runs/final-paper-results/`）

`final_results.json`、`main_by_category.csv`、`main_table.tex`、`robustness_table.tex`、`freeze_manifest.json`。

### 17.5 代码（`reliability_v2/gate/`）

`features.py`、`utility_labels.py`、`gate_model.py`、`infer_gate.py`、`train_gate.py`、`utility_worker.py`、`data_queue.py`、`evaluate_gate.py`（含已修复行）、`audit_phase4_1.py`、`trace_phase4_1.py`、`repair_phase4_1.py`、`statistics.py`、`report_gate.py`、`report_phase4_1.py`、`validate_phase4_1.py`、`validate_phase4.py`。

### 17.6 测试（`tests/`）

`test_gate_feature_schema_parity.py, test_gate_scaler_reuse.py, test_gate_threshold_exact.py, test_gate_sample_key_alignment.py, test_gate_bypass_score_identity.py, test_gate_restore_score_identity.py, test_zero_restore_equals_dark.py, test_phase3_phase4_dark_identity.py, test_phase3_phase4_icra_identity.py, test_gate_probability_recompute.py, test_phase4_1_validator.py`（另含 `test_gate_core.py, test_gate_batches.py, test_gate_data_queue.py`）。

---

## 18. 复现命令

在工作树根 `.`（`MIRNetv2\.worktrees\industrial-lowlight`）运行：

```powershell
# 静态审计 + 根因复现
.venv/Scripts/python.exe -u -m reliability_v2.gate.audit_phase4_1

# 逐样本 trace
.venv/Scripts/python.exe -u -m reliability_v2.gate.trace_phase4_1

# 最小修复后重跑聚合/统计
.venv/Scripts/python.exe -u -m reliability_v2.gate.repair_phase4_1

# 生成主报告
.venv/Scripts/python.exe -u -m reliability_v2.gate.report_phase4_1

# 运行 17 项校验器
.venv/Scripts/python.exe -u -m reliability_v2.gate.validate_phase4_1

# 跑新增测试
.venv/Scripts/python.exe -m pytest tests -k "phase4 or gate or zero_restore"
```

> 说明：Phase 4.1 的审计/修复/校验均为**离线只读重算**，不需要重新生成图像、不需要重训 detector/ICRA。原始数据生成队列 `data_queue` 依赖 GPU，仅需复现原始 Phase 4 时才用。

---

## 19. 当前项目状态与下一步

**实验已封板。** 依据 `工业低照度异常检测论文正式写作 Codex 总提示词.md`，项目已进入正式 SCI 论文写作阶段：

- 已完成：MIRNet-v2 复现、ICRA-MIRNet 低照度适配、MVTec 15 类三随机种子主实验、mild/training_range/severe、MPDD zero-adaptation、标准 PatchCore 验证、Phase 4 label-free Gate、Phase 4.1 一致性审计、聚合 bug 修复、Phase 4 最终真实负结果确认。
- 当前活动：`WileyDesign/Optimal-Design-layout` 下的 Wiley LaTeX 排版（9 月 7 日最新修改）。

**给接手者/后续写作的建议**：

1. 论文结论侧重：Phase 3 的条件性结论为正面贡献（增强收益依赖 severity/方法/类别）；Phase 4 的 label-free Gate 为诚实负结果（normal 域 utility 代理不可迁移），作为 evidence/limitation。
2. 论文表格直接引用 §3 的 Phase 3 冻结主表与 §11 的修复后 Phase 4 主表；原始 `restore_rate=0` 的旧表**勿再作为正式结果引用**。
3. Phase 4.1 validator 已通过，可作为“方法可靠 / 负结果非代码错误”的支撑。

---

_交接文档由当前工作区实验产物整理生成。所有数值均取自冻结/已校验的 JSON 与 CSV，未做任何改动。_
