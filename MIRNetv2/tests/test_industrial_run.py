import torch

from industrial.model import ICRAMIRNet
from industrial.run import gradient_l1, parse_args, restoration_loss, select_balanced_indices, tiled_restore


def test_gradient_l1_is_zero_for_equal_tensors() -> None:
    tensor = torch.rand(1, 3, 8, 8)
    assert gradient_l1(tensor, tensor).item() == 0.0


def test_restoration_loss_backpropagates() -> None:
    restored = torch.rand(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros_like(restored)
    gate = torch.ones(1, 1, 8, 8)
    restoration_loss(restored, target, gate).backward()
    assert restored.grad is not None
    assert torch.isfinite(restored.grad).all()


def test_tiled_restore_handles_narrow_edge_tile() -> None:
    image = torch.rand(1, 3, 128, 132)
    restored = tiled_restore(ICRAMIRNet(torch.nn.Identity()), image, tile_size=128, overlap=16)
    assert restored.shape == image.shape


def test_parse_args_accepts_evaluation_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--data-root", "data", "--weights", "weights.pth", "--max-eval-images", "2"],
    )
    assert parse_args().max_eval_images == 2


def test_evaluation_limit_balances_normal_and_defect_samples() -> None:
    labels = [1, 1, 1, 0, 0, 1, 0, 1]
    selected = select_balanced_indices(labels, max_images=6)
    assert len(selected) == 6
    assert sum(labels[index] == 0 for index in selected) == 3
    assert sum(labels[index] == 1 for index in selected) == 3
