import torch
from torch import nn
from .seed import seed_everything
from .transforms import DbMelSpec

from .train_utils import evaluate
from .dataloaders import load_dataloader
from .configs import Config
from .model import AudioClassifier
import argparse
from pathlib import Path
import json
from safetensors.torch import load_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="path to model folder or json pointer",
    )
    parser.add_argument("--device", "-d", type=str)
    return parser.parse_args()


def load_cfg_model_state(model_path: str, device: str | None):
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


def load_model(cfg: Config) -> nn.Module:
    db_mel_spec = DbMelSpec(cfg).to(cfg.device)
    model = AudioClassifier(len(cfg.subset)).to(cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, db_mel_spec


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = get_args()
    cfg, state_dict = load_cfg_model_state(args.model, args.device)
    model, db_mel_spec = load_model(cfg)

    seed_everything(cfg.seed)

    test_loader = load_dataloader(cfg, "test")
    loss, acc = evaluate(model, cfg, test_loader, db_mel_spec)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
