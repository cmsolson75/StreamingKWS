import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from .configs import Config
from .dataloaders import (
    load_background_loader,
    load_dataloader,
    load_oov_loader,
    load_silence_loader,
    load_speech_cmds,
)
from .model import model_factory
from .seed import seed_everything
from .train_utils import evaluate
from .transforms import DbMelSpec


def get_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="path to checkpoint dir or pointer json (best.json/latest.json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "oov", "kw", "silence", "background"],
        help="which split to evaluate",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="optional device override: cpu|cuda|mps|auto",
    )
    return parser.parse_known_args()


def resolve_checkpoint_dir(model_path: str) -> Path:
    path = Path(model_path)
    if path.suffix == ".json":
        pointer = json.loads(path.read_text())
        path = path.parent / pointer["path"]

    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"model.safetensors missing in: {path}")

    return path


def load_cfg_model_state(
    model_path: str,
    device: str | None = None,
    overrides: list[str] | None = None,
) -> tuple[Config, dict[str, torch.Tensor], Path]:
    ckpt_dir = resolve_checkpoint_dir(model_path)
    base_path = ckpt_dir.parent.parent
    json_config = base_path / "config.resolved.json"
    if not json_config.exists():
        raise FileNotFoundError(f"config.resolved.json missing at: {json_config}")

    merged_overrides = list(overrides or [])
    if device is not None:
        merged_overrides.append(f"train.device={device}")

    cfg = Config.from_json(str(json_config)).with_overrides(merged_overrides)
    state_dict = load_file(ckpt_dir / "model.safetensors")
    return cfg, state_dict, ckpt_dir


def load_model(cfg: Config, state_dict: dict[str, torch.Tensor]) -> nn.Module:
    num_classes = len(cfg.data.subset) + 2
    model = model_factory(cfg.model.name, num_classes, cfg.train.dropout).to(
        cfg.train.device
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_eval_loader(cfg: Config, split: str):
    if split == "test":
        return load_dataloader(cfg, "test")
    if split == "val":
        return load_dataloader(cfg, "val")
    if split == "oov":
        return load_oov_loader(cfg, "val")
    if split == "kw":
        return load_speech_cmds(cfg, "val")
    if split == "silence":
        return load_silence_loader(cfg)
    if split == "background":
        return load_background_loader(cfg)
    raise ValueError(f"unsupported split: {split}")


def main() -> None:
    args, overrides = get_args()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg, state_dict, ckpt_dir = load_cfg_model_state(
        args.model, device=args.device, overrides=overrides
    )
    seed_everything(cfg.env.seed)

    model = load_model(cfg, state_dict)
    db_mel_spec = DbMelSpec(cfg, augment=False).to(cfg.train.device)
    loader = load_eval_loader(cfg, args.split)

    loss, acc = evaluate(model, cfg, loader, db_mel_spec)
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Device: {cfg.train.device} | Split: {args.split}")
    print(f"Loss: {loss:.4f} | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
