"""
Take in a single audio clip that is variable length, but they will be friendly and a single example.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .configs import Config, InferConfig
from .model import AudioClassifier
from .transforms import DbMelSpec, AudioTransform
import json
from pathlib import Path
import soundfile as sf
from safetensors.torch import load_file

from typing import Dict, Tuple


class InferenceRunner(nn.Module):
    def __init__(self, model: AudioClassifier, cfg: Config):
        super().__init__()
        self.model = model
        model.eval()

        self.cfg = cfg

        self.transform = AudioTransform(cfg)
        self.db_mel_spec = DbMelSpec(cfg)

    def forward(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        """
        return probabilities from model not raw logits
        """

        x = self.transform(x, sr)
        x = self.db_mel_spec(x)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        return probs


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
        device_overwrite = [f"device={device}"]
    return Config.from_json(str(json_config)).with_overrides(
        device_overwrite
    ), state_dict


def load_model(cfg: Config, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    model = AudioClassifier(len(cfg.subset)).to(cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def display_probs(probs: torch.Tensor, labels: list):
    import matplotlib.pyplot as plt
    import numpy as np

    ys = probs.detach().cpu().flatten().numpy()
    xs = np.arange(len(ys))
    xs = [labels[x] for x in xs]
    plt.figure(figsize=(10, 6))
    plt.bar(xs, ys)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/infer.yaml")
    parser.add_argument("--input", "-a", type=Path)
    parser.add_argument("--display_probs", action="store_true")
    args, overrides = parser.parse_known_args()
    infer_cfg = InferConfig.from_yaml(args.config).with_overrides(overrides)
    device = infer_cfg.device
    cfg, state_dict = load_cfg_model_state(infer_cfg.model, device)
    model = load_model(cfg, state_dict)

    inference_runner = InferenceRunner(model, cfg).to(device)

    labels = load_labels("labels.json")
    assert args.input.exists()
    audio, sr = sf.read(args.input)
    with torch.no_grad():
        probs = inference_runner(
            torch.tensor(audio, dtype=torch.float32).to(device), sr
        )

    print("Pred:", labels[torch.argmax(probs).item()])
    if args.display_probs:
        display_probs(probs, labels)
