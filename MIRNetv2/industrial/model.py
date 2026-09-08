"""Illumination-conditioned adapter around an image restoration backbone."""

import torch
from torch import Tensor, nn


class ICRAMIRNet(nn.Module):
    """Blend a backbone residual with a spatial low-light-aware gate."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.gate_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 8.0)
        self._forced_gate: float | None = None

    def force_gate(self, value: float | None) -> None:
        """Set a deterministic gate for tests and ablation runs."""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("forced gate must be in [0, 1]")
        self._forced_gate = value

    @staticmethod
    def _luminance(image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        return image[:, :1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if self._forced_gate is None:
            gate = torch.sigmoid(self.gate_net(1.0 - self._luminance(image)))
        else:
            gate = torch.full_like(image[:, :1], self._forced_gate)
        backbone_output = self.backbone(image)
        restored = image + gate * (backbone_output - image)
        return restored, gate
