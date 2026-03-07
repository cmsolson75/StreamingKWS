from __future__ import annotations

import yaml
from typing import Optional, Self, Literal, Annotated, Any
from pathlib import Path
from pydantic import BaseModel, PositiveInt, PositiveFloat, model_validator, Field
import torch
import json
import hashlib
from .model import ModelName


class DataConfig(BaseModel, frozen=True):
    path: str
    subset: list[str] | None = None


class PreprocessConfig(BaseModel, frozen=True):
    sample_rate: PositiveInt = 16_000
    clip_length: PositiveFloat = 1.0
    n_fft: PositiveInt = 400
    hop_len: PositiveInt = 160
    n_mels: PositiveInt = 64


class TrainConfig(BaseModel, frozen=True):
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
    clip_norm: PositiveFloat = 1.0
    warmup_steps: int = 1000
    dropout: Annotated[float, Field(ge=0, le=1)] = 0.5


class EnvConfig(BaseModel, frozen=True):
    seed: PositiveInt = 42
    resume: str | None = None
    remote_name: str | None = None
    cloud_sync: bool = False


class SamplerConfig(BaseModel, frozen=True):
    keyword_weight: Annotated[float, Field(ge=0, le=1)] = 0.7
    oov_weight: Annotated[float, Field(ge=0, le=1)] = 0.2
    silence_weight: Annotated[float, Field(ge=0, le=1)] = 0.05
    background_weight: Annotated[float, Field(ge=0, le=1)] = 0.05

    @model_validator(mode="after")
    def check_sum(self) -> Self:
        current_sum = round(
            self.keyword_weight
            + self.oov_weight
            + self.silence_weight
            + self.background_weight,
            4,
        )
        if current_sum != 1.0:
            raise ValueError(f"Sampler probs don't add up to 1.0, got {current_sum}")
        return self


class AugmentConfig(BaseModel, frozen=True):
    use_augmentations: bool = False
    noise_path: str
    prob_noise_mix_in: Annotated[float, Field(ge=0, le=1)] = 0.75
    min_snr_db: float = -5.0
    max_snr_db: float = 25.0
    prob_gain_aug: float = 0.75
    min_db_gain: float = -15.0
    max_db_gain: float = 5.0
    freq_mask_param: int = 15
    time_mask_param: int = 35


# ModelName = Literal["cnn", "tc_resnet8"]


class ModelConfig(BaseModel, frozen=True):
    name: ModelName


class Config(BaseModel, frozen=True):
    data: DataConfig
    preprocess: PreprocessConfig = PreprocessConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig
    env: EnvConfig = EnvConfig()
    augment: AugmentConfig | None = None
    sampler: SamplerConfig = SamplerConfig()

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
            raw = json.load(f)
        return cls.model_validate(raw)

    def hash(self, length: int = 10) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]

    def to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @model_validator(mode="after")
    def resolve_auto_device(self) -> Self:
        if self.train.device == "auto":
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
            train_update = {"device": device}
            if device == "cuda":
                update_dict["pin_memory"] = True

            return self.model_copy(
                update={"train": self.train.model_copy(update=train_update)}
            )
        return self

    def _set_by_dotted_key(self, d: dict[str, Any], key: str, value: Any) -> None:
        parts = key.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def with_overrides(self, overrides: list[str]) -> Self:
        if not overrides:
            return self
        updates = self.model_dump()

        for ov in overrides:
            k, v = ov.split("=", 1)
            self._set_by_dotted_key(updates, k, yaml.safe_load(v))
        return self.model_validate(updates)


class InferConfig(BaseModel, frozen=True):
    model: str
    label_file: str
    device: str
    wakeword: str = "marvin"
    stopword: str = "stop"
    threshold: float = 0.7
    cooldown_time: float = 1.0
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 16000
    number_map: dict[str, int] = Field(
        default_factory=lambda: {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
        }
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        cfg_path = Path(path)
        with cfg_path.open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

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
