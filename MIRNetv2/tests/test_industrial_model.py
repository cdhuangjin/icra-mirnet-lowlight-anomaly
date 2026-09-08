import torch

from industrial.model import ICRAMIRNet


def test_gate_one_reproduces_identity_backbone() -> None:
    model = ICRAMIRNet(torch.nn.Identity())
    model.force_gate(1.0)
    image = torch.rand(1, 3, 8, 8)
    restored, gate = model(image)
    assert torch.allclose(restored, image)
    assert torch.allclose(gate, torch.ones_like(gate))


def test_gate_matches_input_spatial_shape() -> None:
    restored, gate = ICRAMIRNet(torch.nn.Identity())(torch.rand(2, 3, 16, 12))
    assert restored.shape == (2, 3, 16, 12)
    assert gate.shape == (2, 1, 16, 12)
