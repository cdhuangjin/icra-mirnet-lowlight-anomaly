import json
from pathlib import Path


def test_full_config_is_single_gpu_and_seeded() -> None:
    config = json.loads(Path("industrial/configs/mvtec_full.json").read_text(encoding="utf-8"))
    assert config["batch_size"] == 2
    assert config["accumulation_steps"] == 4
    assert config["seed"] == 20260830
    assert len(config["categories"]) == 15
