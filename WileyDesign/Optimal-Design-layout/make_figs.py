import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from matplotlib.ticker import FormatStrFormatter

OUT = r'C:\Users\PC\Documents\Codex\Reproduction003\WileyDesign\Optimal-Design-layout\images'

# --- Figure 1: clear / low-light / ICRA-restored sample (MVTec bottle broken_large) ---
clear = r'D:\data数据集\MVTec AD\bottle\test\broken_large\000.png'
dark = r'C:\Users\PC\Documents\Codex\Reproduction003\MIRNetv2\.worktrees\industrial-lowlight\runs\reliability_v2\enhancement_outputs\dark\training_range\bottle\broken_large\000.png'
icra = r'C:\Users\PC\Documents\Codex\Reproduction003\MIRNetv2\.worktrees\industrial-lowlight\runs\reliability_v2\enhancement_outputs\icra\training_range\bottle\broken_large\000.png'

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=300)
titles = ['Clear', 'Low-light (training-range)', 'ICRA-MIRNet restorer']
for ax, path, t in zip(axes, [clear, dark, icra], titles):
    img = imread(path)
    ax.imshow(img)
    ax.set_title(t, fontsize=7)
    ax.axis('off')
plt.tight_layout(pad=0.3)
plt.savefig(OUT + r'\fig_sample.pdf')
plt.close()

# --- Figure 2: class-macro anomaly AUROC by severity and domain ---
severities = ['mild', 'training_range', 'severe']
dark_auroc = [0.944966, 0.799974, 0.756876]
mirnet_auroc = [0.928972, 0.898658, 0.775847]
icra_auroc = [0.962859, 0.937793, 0.775843]
gate_auroc = [0.963588, 0.880869, 0.749202]

x = np.arange(len(severities))
w = 0.2
fig, ax = plt.subplots(figsize=(5.4, 3.1), dpi=300)
ax.bar(x - 1.5*w, dark_auroc, w, label='Dark', color='#8c8c8c')
ax.bar(x - 0.5*w, mirnet_auroc, w, label='MIRNet-v2', color='#4c8bf5')
ax.bar(x + 0.5*w, icra_auroc, w, label='ICRA-MIRNet', color='#2f9e6e')
ax.bar(x + 1.5*w, gate_auroc, w, label='Gate (label-free)', color='#d9822b')
ax.set_xticks(x)
ax.set_xticklabels(['mild', 'training_range', 'severe'])
ax.set_ylabel('Class-macro AUROC')
ax.set_ylim(0.7, 1.0)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(fontsize=6.5, frameon=False)
ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(OUT + r'\fig_auroc.pdf')
plt.close()

print('FIGURES_WRITTEN')
