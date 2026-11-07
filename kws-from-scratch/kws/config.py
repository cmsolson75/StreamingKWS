"""M1 · 配置系统

目标
----
把 configs/config.yaml 变成一个**有类型、会校验、不可变**的 Python 对象，
并支持命令行覆盖：`--config configs/config.yaml train.batch_size=32 augment.enabled=false`。

为什么要这样做
--------------
- 有类型：`cfg.train.lr` 有补全、能被 mypy 检查，拼错字段立刻报错，而不是训练到一半才发现。
- 会校验：采样权重加起来不是 1、SNR 下限大于上限……这类错误要在启动时就拦住。
- 不可变（frozen）：训练中途没人能偷偷改配置，所以存下来的配置就是真正用过的配置。
- 哈希：同样的配置得到同样的哈希，可以用来给 run 目录命名、判断两个实验是否同配置。

先读文档
--------
- Pydantic Models：https://docs.pydantic.dev/latest/concepts/models/
  重点：`BaseModel`、`model_config` / 类参数 `frozen=True`、`extra="forbid"`、
  `model_dump`、`model_validate`、`model_dump_json`
- Pydantic Validators：https://docs.pydantic.dev/latest/concepts/validators/
  重点：`model_validator(mode="after")`
- PyYAML：`yaml.safe_load`（思考：为什么不用 `yaml.load`？）

思考题（先写进学习日志，再写代码）
---------------------------------
1. `PreprocessConfig` 是 frozen 的。那 `with_overrides` 怎么“修改”配置？返回的是同一个对象吗？
2. 覆盖 `train.batch_size=32` 时，命令行给你的是字符串 "32"。谁负责把它变成 int？
   `env.resume=null`、`data.keywords=[a,b]` 又分别会变成什么？
3. 为什么覆盖之后必须**重新校验整个配置**，而不是只改那一个字段？
   （举一个“单个字段合法、组合起来不合法”的例子。）
4. 哈希要“稳定”：同一个配置在两台机器、两次运行里得到同一个值。
   如果直接用 Python 内置的 `hash()` 会怎样？dict 的键顺序会影响结果吗？

写完的标准：`uv run pytest tests/test_m01_infra.py -k config` 全绿。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# 示范：这一个已经写好了，照它的模式写下面其它几个。
# 注意三件事：frozen、extra="forbid"、每个字段都有类型。
# ---------------------------------------------------------------------------
class PreprocessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_rate: int = 16000
    clip_seconds: float = 1.0
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 64

    @property
    def num_samples(self) -> int:
        """一个片段有多少个采样点。属性（property）不是字段，不会出现在 model_dump 里。"""
        return int(self.sample_rate * self.clip_seconds)


class DataConfig(BaseModel):
    # TODO(M1): 字段对照 config.yaml 的 data 段：root（字符串）、keywords（字符串列表）
    model_config = ConfigDict(frozen=True, extra="forbid")


class SamplerConfig(BaseModel):
    # TODO(M1): 四个 float 字段：keyword、unknown、silence、background
    # TODO(M1): 用 model_validator(mode="after") 检查：每个权重 >= 0，并且总和约等于 1。
    #           “约等于”是多少？为什么不能用 == 1.0？
    model_config = ConfigDict(frozen=True, extra="forbid")


class AugmentConfig(BaseModel):
    # TODO(M1): 字段对照 config.yaml 的 augment 段。
    # TODO(M1): 校验：两个概率在 [0, 1]；min_snr_db <= max_snr_db；min_gain_db <= max_gain_db。
    #           提示：概率范围可以用 Field 的约束参数，不一定要写在 validator 里。
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(BaseModel):
    # TODO(M1): name 只能是 "cnn"、"tc_resnet8"、"transformer" 三者之一。用 Literal。
    #           dropout、width_mult 两个 float。
    model_config = ConfigDict(frozen=True, extra="forbid")


class TrainConfig(BaseModel):
    # TODO(M1): 字段对照 config.yaml 的 train 段。
    #           device 只能是 "auto" | "cpu" | "cuda" | "mps"。
    #           校验：warmup_steps <= max_steps。
    model_config = ConfigDict(frozen=True, extra="forbid")


class EnvConfig(BaseModel):
    # TODO(M1): seed（int）、run_root（str）、resume（str 或 None）
    model_config = ConfigDict(frozen=True, extra="forbid")


Device = Literal["cpu", "cuda", "mps"]


class Config(BaseModel):
    """顶层配置。字段名必须和 YAML 的顶层键一一对应。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # TODO(M1): data、preprocess、sampler、augment、model、train、env 七个子配置

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """读 YAML 文件，返回校验过的 Config。"""
        raise NotImplementedError("M1: Config.from_yaml")

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """读 to_json 写出的文件。恢复训练时要靠它还原当时的配置。"""
        raise NotImplementedError("M1: Config.from_json")

    def to_json(self, path: str | Path) -> None:
        """写成带缩进的 JSON。要求：from_json(to_json(cfg)) == cfg。"""
        raise NotImplementedError("M1: Config.to_json")

    def with_overrides(self, overrides: list[str]) -> Self:
        """应用形如 "a.b=value" 的覆盖，返回一个**新的**、重新校验过的 Config。

        要求：
        - value 按 YAML 规则解析（"32" → 32，"false" → False，"null" → None，"[a,b]" → 列表）
        - 键不存在时报错（KeyError 或 ValueError），不能悄悄新增字段
        - 格式不是 "key=value" 时报 ValueError
        - 原来的 self 不变

        步骤（只给方向）：变成普通 dict → 按点号一层层往下走 → 改值 → 重新构造 Config。
        """
        raise NotImplementedError("M1: Config.with_overrides")

    def hash(self, length: int = 10) -> str:
        """配置的稳定哈希，返回 length 个十六进制字符。

        要求：同样的配置 → 同样的哈希（跨进程、跨机器）；任何字段变化 → 哈希变化。
        相关 API：hashlib、json.dumps 的 sort_keys 参数。
        """
        raise NotImplementedError("M1: Config.hash")

    def resolved_device(self) -> Device:
        """把 train.device="auto" 解析成实际设备：有 CUDA 用 cuda，其次 mps，否则 cpu。
        不是 auto 时原样返回。相关 API：torch.cuda.is_available、torch.backends.mps.is_available
        """
        raise NotImplementedError("M1: Config.resolved_device")
