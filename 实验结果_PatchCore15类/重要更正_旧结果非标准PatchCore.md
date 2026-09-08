# 旧结果方法名称更正

2026-09-05 代码复核确认：本目录此前提供的 CSV/报告来自 ResNet18 全局平均池化特征最近邻探针，并非标准局部 patch memory-bank PatchCore。

旧数值和文件全部保留用于审计，不得引用为标准 PatchCore 正式结果；旧 validator PASS 仅代表当时覆盖的数据和数值检查通过，并未验证算法身份。

经用户确认，现使用 Amazon Science 官方 PatchCore 源码重新运行 bottle 验证及 MVTec AD 15 类 mild/training_range/severe。新结果与旧结果隔离：

`C:\Users\PC\Documents\Codex\Reproduction003\MIRNetv2\.worktrees\industrial-lowlight\runs\reliability_v2\patchcore_official`

只有全矩阵完成且通过新验证后，才可报告正式三强度结果。目前请以新目录日志和逐类 completion.json 判断实际进度。
