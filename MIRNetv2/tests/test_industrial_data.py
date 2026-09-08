from pathlib import Path

import cv2
import numpy as np
import torch

from industrial.data import LowLightDegrader, MVTecLowLightDataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((12, 12, 3), 128, dtype=np.uint8))


def test_fixed_seed_is_reproducible() -> None:
    image = torch.full((3, 16, 16), 0.5)
    first = LowLightDegrader(seed=7)(image, index=3)
    second = LowLightDegrader(seed=7)(image, index=3)
    assert torch.equal(first, second)


def test_train_indexes_only_good_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "bottle" / "train" / "good" / "train.png")
    _write_image(tmp_path / "bottle" / "test" / "broken_large" / "test.png")
    dataset = MVTecLowLightDataset(
        root=tmp_path, categories=["bottle"], phase="train", patch_size=8, seed=3
    )
    assert len(dataset) == 1
    assert "train/good" in dataset.items[0].clean_path.as_posix()


def test_test_labels_good_and_defect_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "bottle" / "test" / "good" / "good.png")
    _write_image(tmp_path / "bottle" / "test" / "broken_large" / "bad.png")
    dataset = MVTecLowLightDataset(
        root=tmp_path, categories=["bottle"], phase="test", patch_size=None, seed=3
    )
    assert [item.label for item in dataset.items] == [1, 0]


def test_reader_supports_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "数据" / "image.png"
    path.parent.mkdir()
    encoded = cv2.imencode(".png", np.full((8, 8, 3), 128, dtype=np.uint8))[1]
    encoded.tofile(str(path))
    image = MVTecLowLightDataset._read_rgb(path)
    assert image.shape == (3, 8, 8)
