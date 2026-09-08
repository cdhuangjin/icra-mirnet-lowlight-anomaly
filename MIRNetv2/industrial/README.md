# 工业巡检低照度增强实验

本目录实现 ICRA-MIRNet：在官方 MIRNet-v2 输出残差上使用逐像素照度门控，使暗区获得更强校正、明亮区域保持稳定。数据使用本地 MVTec AD；清晰图经确定性 gamma、空间照度、泊松噪声和读出噪声退化后作为输入。

## 环境

使用已有隔离环境：

```powershell
& 'C:\Users\PC\Documents\Codex\Reproduction003\MIRNetv2\.venv\Scripts\python.exe' -m pytest tests -q --basetemp .pytest_tmp
```

## 复现命令

从工作树根目录运行 smoke：

```powershell
& 'C:\Users\PC\Documents\Codex\Reproduction003\MIRNetv2\.venv\Scripts\python.exe' -m industrial.run --config industrial/configs/bottle_smoke.json
```

该配置使用两个固定测试图（一个正常、一个缺陷），仅验证端到端流程，不能报告为研究结论。

单类先导实验（bottle；200 步、20 正常 + 20 缺陷测试图）：

```powershell
& 'C:\Users\PC\Documents\Codex\Reproduction003\MIRNetv2\.venv\Scripts\python.exe' -m industrial.run --config industrial/configs/bottle_pilot.json
```

该先导结果用于确定训练和评估协议可行，不能替代全 15 类报告。

对应的预训练但未在 MVTec 微调的对照：`industrial/configs/bottle_pretrained.json`。

完整 MVTec 实验：

```powershell
& 'C:\Users\PC\Documents\Codex\Reproduction003\MIRNetv2\.venv\Scripts\python.exe' -m industrial.run --config industrial/configs/mvtec_full.json
```

输出目录包含模型 checkpoint 与 `metrics.json`。报告中应同时给出清晰、低照度、增强后三种输入的 image-level AUROC，及 PSNR、SSIM、延迟和峰值显存。只有在低照度输入上提升恢复指标且异常检测不退化时，才能声称对巡检有帮助。

## 数据限制

本实验使用**合成**低照度；MVTec AD 不是真实夜间工业相机数据。结果只证明对固定合成退化协议的有效性，不能外推为真实夜间工厂性能。论文必须披露这个限制，并将配对真实低照度采集列为后续工作。
