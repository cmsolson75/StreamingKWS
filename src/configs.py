import yaml
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Self, Dict
from pathlib import Path


@dataclass(frozen=True)
class Config:
    path: str
    subset: Optional[List[str]] = None
    # preprocess
    sample_rate: int = 16_000
    clip_length: float = 1.0
    n_fft: int = 400
    hop_len: int = 160
    n_mels: int = 64
    # training
    learning_rate: float = 3e-3
    batch_size: int = 64
    num_workers: int = 0
    persistant_workers: bool = False
    prefetch_factor: Optional[int] = None
    pin_memory: bool = False
    device: str = "cpu"
    max_steps: int = 20_000
    amp: bool = False
    log_period: int = 200
    eval_period: int = 2000

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        cfg_path = Path(path)
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)


if __name__ == "__main__":
    # import argparse
    # from pathlib import Path

    # parser = argparse.ArgumentParser()
    # parser.add_argument("path")
    # args = parser.parse_args()
    # dataset_config = DatasetConfig.from_yaml(args.path)
    # print(dataset_config)
    print(Config.from_yaml("configs/config.yaml"))
