import yaml
from typing import Optional, List, Self, Literal
from pathlib import Path
from pydantic import BaseModel, PositiveInt, PositiveFloat, model_validator
import torch
import json
import hashlib


class Config(BaseModel, frozen=True):
    path: str
    subset: Optional[List[str]] = None
    # preprocess
    sample_rate: PositiveInt = 16_000
    clip_length: PositiveFloat = 1.0
    n_fft: PositiveInt = 400
    hop_len: PositiveInt = 160
    n_mels: PositiveInt = 64
    # training
    learning_rate: PositiveFloat = 3e-3
    batch_size: PositiveInt = 64
    num_workers: int = 0
    persistent_workers: bool = False
    prefetch_factor: Optional[PositiveInt] = None
    pin_memory: bool = False
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    max_steps: PositiveInt = 20_000
    amp: bool = False
    log_period: PositiveInt = 200
    eval_period: PositiveInt = 2000

    # env
    seed: PositiveInt = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        cfg_path = Path(path)
        with cfg_path.open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        cfg_path = Path(path)
        with cfg_path.open("r") as f:
            raw = json.loads(f)
        return cls.model_validate(raw)

    def hash(self, length: int = 10) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]

    def to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def with_overrides(self, overrides: list[str]) -> Self:
        if not overrides:
            return self

        updates = self.model_dump()

        for ov in overrides:
            key, val = ov.split("=", 1)
            val = yaml.safe_load(val)
            parts = key.split(".")

            d = updates
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = val
        return self.model_validate(updates)

    @model_validator(mode="after")
    def resolve_auto_device(self) -> Self:
        if self.device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            update_dict = {
                "device": device,
            }
            # auto set pin memory for CUDA
            if device == "cuda":
                update_dict["pin_memory"] = True

            return self.model_copy(update={"device": device})
        return self


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/config.yaml")
    args, overwrites = parser.parse_known_args()
    cfg = Config.from_yaml(args.config).with_overrides(overwrites)
    print(cfg)
