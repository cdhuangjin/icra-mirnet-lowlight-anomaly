"""Train and evaluate ICRA-MIRNet on deterministic low-light MVTec AD data."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from basicsr.models.archs.mirnet_v2_arch import MIRNet_v2
from industrial.data import MVTecLowLightDataset
from industrial.model import ICRAMIRNet


def gradient_l1(restored: Tensor, target: Tensor) -> Tensor:
    """L1 distance between horizontal and vertical image gradients."""
    restored_x, target_x = restored[..., :, 1:] - restored[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    restored_y, target_y = restored[..., 1:, :] - restored[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(restored_x, target_x) + F.l1_loss(restored_y, target_y)


def restoration_loss(restored: Tensor, target: Tensor, gate: Tensor, gradient_weight: float = 0.2, gate_weight: float = 0.05) -> Tensor:
    luminance = target[:, :1] * 0.299 + target[:, 1:2] * 0.587 + target[:, 2:3] * 0.114
    gate_target = 1.0 - luminance
    return F.l1_loss(restored, target) + gradient_weight * gradient_l1(restored, target) + gate_weight * F.l1_loss(gate, gate_target)


def select_balanced_indices(labels: list[int], max_images: int) -> list[int]:
    """Select a deterministic, near-balanced normal/defect evaluation subset."""
    if max_images < 2:
        raise ValueError("max_images must be at least 2 to compute AUROC")
    groups = {
        0: [index for index, label in enumerate(labels) if label == 0],
        1: [index for index, label in enumerate(labels) if label == 1],
    }
    if not groups[0] or not groups[1]:
        raise ValueError("test set must include both good and defect images")
    per_group = max_images // 2
    selected = groups[0][:per_group] + groups[1][:per_group]
    remaining = max_images - len(selected)
    if remaining:
        leftovers = groups[0][per_group:] + groups[1][per_group:]
        selected.extend(leftovers[:remaining])
    return selected


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_backbone(weights_path: str | Path) -> MIRNet_v2:
    backbone = MIRNet_v2(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=1)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(checkpoint["params"], strict=True)
    return backbone


def tiled_restore(model: ICRAMIRNet, image: Tensor, tile_size: int = 128, overlap: int = 16) -> Tensor:
    """Restore a single image with overlap averaging to retain the 8 GB limit."""
    _, _, height, width = image.shape
    stride = tile_size - overlap
    output = torch.zeros_like(image)
    weight = torch.zeros_like(image[:, :1])
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            tile = image[:, :, top:bottom, left:right]
            pad_h, pad_w = tile_size - tile.shape[-2], tile_size - tile.shape[-1]
            if pad_h or pad_w:
                mode = "reflect" if pad_h < tile.shape[-2] and pad_w < tile.shape[-1] else "replicate"
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode=mode)
            restored, _ = model(tile)
            restored = restored[:, :, : bottom - top, : right - left]
            output[:, :, top:bottom, left:right] += restored
            weight[:, :, top:bottom, left:right] += 1
    return output / weight


class FeatureGaussian:
    """Diagonal Gaussian on frozen ImageNet features for image-level anomaly scores."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        try:
            encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as error:  # keep the experiment auditable instead of silently using random features
            raise RuntimeError("ImageNet ResNet-18 weights are required for anomaly evaluation") from error
        self.encoder = nn.Sequential(*list(encoder.children())[:-1]).to(device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.mean: Tensor | None = None
        self.variance: Tensor | None = None

    @torch.inference_mode()
    def _features(self, images: list[Tensor]) -> Tensor:
        batches: list[Tensor] = []
        for image in images:
            resized = F.interpolate(image.unsqueeze(0).to(self.device), size=(224, 224), mode="bilinear", align_corners=False)
            batches.append(self.encoder(resized).flatten(1).cpu())
        return torch.cat(batches, dim=0)

    def fit(self, images: list[Tensor]) -> None:
        features = self._features(images)
        self.mean = features.mean(dim=0)
        self.variance = features.var(dim=0, unbiased=False).clamp_min(1e-6)

    def score(self, images: list[Tensor]) -> np.ndarray:
        if self.mean is None or self.variance is None:
            raise RuntimeError("fit must be called before score")
        features = self._features(images)
        return (((features - self.mean) ** 2) / self.variance).mean(dim=1).numpy()


def image_metrics(restored: Tensor, target: Tensor) -> tuple[float, float]:
    restored_np = restored.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    target_np = target.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    return peak_signal_noise_ratio(target_np, restored_np, data_range=1.0), structural_similarity(target_np, restored_np, channel_axis=2, data_range=1.0)


def train(model: ICRAMIRNet, loader: DataLoader, device: torch.device, steps: int, accumulation_steps: int) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        dark, clean = batch["dark"].to(device), batch["clean"].to(device)
        with autocast(device.type, enabled=device.type == "cuda"):
            restored, gate = model(dark)
            loss = restoration_loss(restored, clean, gate) / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


@torch.inference_mode()
def evaluate(
    model: ICRAMIRNet,
    root: str | Path,
    categories: list[str],
    seed: int,
    device: torch.device,
    max_images: int | None = None,
) -> dict[str, Any]:
    test_data = MVTecLowLightDataset(root, categories, phase="test", patch_size=None, seed=seed)
    train_data = MVTecLowLightDataset(root, categories, phase="train", patch_size=None, seed=seed)
    detector = FeatureGaussian(device)
    detector.fit([train_data[index]["clean"] for index in range(len(train_data))])

    clean_images, dark_images, restored_images, labels = [], [], [], []
    psnrs, ssims, elapsed = [], [], []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    model.eval()
    indices = list(range(len(test_data)))
    if max_images is not None:
        indices = select_balanced_indices([item.label for item in test_data.items], max_images)
    for index in indices:
        item = test_data[index]
        clean, dark = item["clean"], item["dark"]
        started = time.perf_counter()
        restored = tiled_restore(model, dark.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)
        elapsed.append(time.perf_counter() - started)
        psnr, ssim = image_metrics(restored, clean)
        psnrs.append(psnr)
        ssims.append(ssim)
        clean_images.append(clean)
        dark_images.append(dark)
        restored_images.append(restored)
        labels.append(int(item["label"]))

    def auc(images: list[Tensor]) -> float:
        return float(roc_auc_score(labels, detector.score(images)))

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "latency_seconds": float(np.mean(elapsed)),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
        "auroc_clean": auc(clean_images),
        "auroc_dark": auc(dark_images),
        "auroc_restored": auc(restored_images),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--categories", nargs="+", default=["bottle"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-eval-images", type=int)
    parser.add_argument("--output", type=Path, default=Path("runs/industrial"))
    args = parser.parse_args()
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        for key, value in config.items():
            if key == "data_root" or key == "weights" or key == "output":
                value = Path(value)
            setattr(args, key, value)
    if args.data_root is None or args.weights is None:
        parser.error("--data-root and --weights are required unless supplied in --config")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = MVTecLowLightDataset(args.data_root, args.categories, phase="train", patch_size=args.patch_size, seed=args.seed)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    model = ICRAMIRNet(make_backbone(args.weights)).to(device)
    train(model, loader, device, args.steps, args.accumulation_steps)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.output / "icra_mirnet.pth")
    metrics = evaluate(model, args.data_root, args.categories, args.seed, device, args.max_eval_images)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
