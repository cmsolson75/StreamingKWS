import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file
from torch import nn

from ..configs import Config
from ..model import model_factory


def load_labels(label_path: str):
    path = Path(label_path)
    with path.open("r") as f:
        return json.load(f)


def load_cfg_model_state(
    model_path: str, device: str | None = None
) -> Tuple[Config, Dict[str, torch.Tensor]]:
    path = Path(model_path)
    if path.suffix == ".json":
        json_data = json.loads(path.read_text())["path"]
        path = path.parent / json_data
    base_path = path.parent.parent

    state_dict = load_file(path / "model.safetensors")
    json_config = base_path / "config.resolved.json"
    device_overwrite = None
    if device is not None:
        device_overwrite = [f"train.device={device}"]  # was "device=..."
    return Config.from_json(str(json_config)).with_overrides(
        device_overwrite
    ), state_dict


def load_model(cfg: Config, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    num_classes = len(cfg.data.subset) + 2
    model = model_factory(cfg.model.name, num_classes, cfg.train.dropout).to(
        cfg.train.device
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model
