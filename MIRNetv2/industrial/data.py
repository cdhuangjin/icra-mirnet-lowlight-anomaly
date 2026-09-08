"""Deterministic synthetic low-light data for MVTec AD experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MVTecItem:
    clean_path: Path
    label: int
    category: str


class LowLightDegrader:
    """Applies a deterministic, index-addressable low-light camera degradation."""

    def __init__(
        self,
        seed: int,
        gamma: tuple[float, float] = (2.0, 4.0),
        read_noise: tuple[float, float] = (0.0, 0.02),
    ) -> None:
        self.seed = seed
        self.gamma = gamma
        self.read_noise = read_noise

    def _generator(self, index: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(self.seed + index)

    @staticmethod
    def _sample_range(bounds: tuple[float, float], generator: torch.Generator) -> float:
        return torch.empty((), dtype=torch.float32).uniform_(*bounds, generator=generator).item()

    @staticmethod
    def _illumination(height: int, width: int, generator: torch.Generator) -> Tensor:
        y = torch.linspace(0.0, 1.0, height).view(height, 1)
        x = torch.linspace(0.0, 1.0, width).view(1, width)
        slope_x = torch.empty((), dtype=torch.float32).uniform_(-0.30, 0.30, generator=generator)
        slope_y = torch.empty((), dtype=torch.float32).uniform_(-0.30, 0.30, generator=generator)
        offset = torch.empty((), dtype=torch.float32).uniform_(0.55, 0.85, generator=generator)
        return (offset + slope_x * (x - 0.5) + slope_y * (y - 0.5)).clamp(0.35, 1.0)

    def __call__(self, image: Tensor, index: int) -> Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must have shape [3, H, W]")
        generator = self._generator(index)
        gamma = self._sample_range(self.gamma, generator)
        clean = image.detach().cpu().float().clamp(0, 1)
        illumination = self._illumination(clean.shape[1], clean.shape[2], generator).unsqueeze(0)
        dark = clean.pow(gamma) * illumination
        photon_counts = torch.poisson((dark * 255.0).clamp_min(0), generator=generator) / 255.0
        sigma = self._sample_range(self.read_noise, generator)
        read = torch.randn(photon_counts.shape, generator=generator) * sigma
        return (photon_counts + read).clamp(0, 1)


class MVTecLowLightDataset(Dataset[dict[str, Tensor | str | int]]):
    """MVTec clean images paired with generated low-light observations."""

    def __init__(
        self,
        root: str | Path,
        categories: Iterable[str],
        phase: Literal["train", "test"],
        patch_size: int | None,
        seed: int,
    ) -> None:
        self.root = Path(root)
        self.phase = phase
        self.patch_size = patch_size
        self.seed = seed
        self.degrader = LowLightDegrader(seed)
        self.items = self._index(tuple(categories))
        if not self.items:
            raise ValueError(f"no MVTec images found under {self.root}")

    def _index(self, categories: tuple[str, ...]) -> list[MVTecItem]:
        items: list[MVTecItem] = []
        for category in sorted(categories):
            category_root = self.root / category
            if self.phase == "train":
                for path in sorted((category_root / "train" / "good").glob("*")):
                    if path.is_file():
                        items.append(MVTecItem(path, 0, category))
                continue
            for kind_dir in sorted((category_root / "test").iterdir()):
                if not kind_dir.is_dir():
                    continue
                label = 0 if kind_dir.name == "good" else 1
                for path in sorted(kind_dir.glob("*")):
                    if path.is_file():
                        items.append(MVTecItem(path, label, category))
        return items

    @staticmethod
    def _read_rgb(path: Path) -> Tensor:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)

    def _crop(self, clean: Tensor, index: int) -> Tensor:
        if self.patch_size is None:
            return clean
        height, width = clean.shape[-2:]
        pad_h = max(0, self.patch_size - height)
        pad_w = max(0, self.patch_size - width)
        if pad_h or pad_w:
            clean = torch.nn.functional.pad(clean, (0, pad_w, 0, pad_h), mode="reflect")
        generator = torch.Generator(device="cpu").manual_seed(self.seed * 100_003 + index)
        height, width = clean.shape[-2:]
        top = torch.randint(0, height - self.patch_size + 1, (), generator=generator).item()
        left = torch.randint(0, width - self.patch_size + 1, (), generator=generator).item()
        return clean[:, top : top + self.patch_size, left : left + self.patch_size]

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        item = self.items[index]
        clean = self._crop(self._read_rgb(item.clean_path), index)
        return {
            "clean": clean,
            "dark": self.degrader(clean, index),
            "label": item.label,
            "path": str(item.clean_path),
            "category": item.category,
        }

    def __len__(self) -> int:
        return len(self.items)
