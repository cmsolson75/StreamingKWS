# 学习目标：把 YAML / JSON 里的设置变成可以用 cfg.train.batch_size 访问的对象。
# 阅读路线：各小配置类规定格式 → Config 汇总并读取 → with_overrides 应用命令行修改。
# 本文件使用 Pydantic：既保存设置，也在创建配置时检查必填项、类型和部分数值范围。
from __future__ import annotations

import yaml
from typing import Optional, Self, Literal, Annotated, Any
from pathlib import Path
from pydantic import BaseModel, PositiveInt, PositiveFloat, model_validator, Field
import torch
import json
import hashlib
from .model import ModelName


# ======================================================================
# 1. 按用途声明设置：字段名、类型和默认值
# ======================================================================
# BaseModel 是 Pydantic 的配置基类；没有 = 默认值的字段创建时必须提供。
# frozen=True 禁止直接重新赋值字段；修改配置时通常创建一个新对象。
# PositiveInt / PositiveFloat 要求大于 0；list[str] | None 表示字符串列表或空值。
class DataConfig(BaseModel, frozen=True):
    path: str
    subset: list[str] | None = None


# 音频格式：sample_rate 每秒采样点数，clip_length 时长，n_fft 窗口变换长度，hop_len 移动步长。
# n_mels 是最终频谱的频率行数，需要与模型输入一致。
class PreprocessConfig(BaseModel, frozen=True):
    sample_rate: PositiveInt = 16_000
    clip_length: PositiveFloat = 1.0
    n_fft: PositiveInt = 400
    hop_len: PositiveInt = 160
    n_mels: PositiveInt = 64


# 训练：学习率和每批样本数决定怎样更新；workers 等决定加载方式，device 决定计算设备。
# log_period / eval_period 是间隔多少步记录 / 验证；amp 是混合精度开关。
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
    # Annotated 给类型补充限制；Field(ge=0, le=1) 要求值位于闭区间 [0, 1]。
    dropout: Annotated[float, Field(ge=0, le=1)] = 0.5


# 实验环境：seed 固定随机起点，resume 指定旧实验，cloud_sync 决定是否启用云同步。
class EnvConfig(BaseModel, frozen=True):
    seed: PositiveInt = 42
    resume: str | None = None
    remote_name: str | None = None
    cloud_sync: bool = False


# 抽样比例：关键词、非目标词、静音、背景噪声四类，下面还会检查总和。
class SamplerConfig(BaseModel, frozen=True):
    keyword_weight: Annotated[float, Field(ge=0, le=1)] = 0.7
    oov_weight: Annotated[float, Field(ge=0, le=1)] = 0.2
    silence_weight: Annotated[float, Field(ge=0, le=1)] = 0.05
    background_weight: Annotated[float, Field(ge=0, le=1)] = 0.05

    @model_validator(mode="after")
    def check_sum(self) -> Self:
        # 装饰器让 Pydantic 在读完字段后自动检查；Self 表示返回同类配置对象。
        # round(..., 4) 先保留四位小数，减少浮点加法的小误差，再检查是否等于 1。
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


# 增强：概率决定是否执行，SNR 范围决定噪声强弱，gain 范围决定音量变化。
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


# ModelName 从 model.py 导入，约定可选名称，避免两处各自维护一份。


class ModelConfig(BaseModel, frozen=True):
    name: ModelName


# ======================================================================
# 2. 汇总训练配置：读取文件、保存结果、选择设备、覆盖参数
# ======================================================================
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
        # 输入文件路径，输出 Config；classmethod 让调用方可以直接写 Config.from_yaml(...)。
        # cls 指当前类；safe_load 将 YAML 读成普通字典，model_validate 再创建并校验配置。
        cfg_path = Path(path)
        with cfg_path.open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        # JSON 版本同理；json.load 读文件，区别于 json.loads 读取一段 JSON 字符串。
        cfg_path = Path(path)
        with cfg_path.open("r") as f:
            raw = json.load(f)
        return cls.model_validate(raw)

    def hash(self, length: int = 10) -> str:
        # model_dump() 转成普通字典；按固定顺序序列化，再用 sha256 生成配置摘要。
        # 为什么排序：内容一样、键排列不同的配置也应得到相同摘要，便于标识实验设置。
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]

    def to_json(self, path: Path) -> None:
        # 把实际生效的设置保存到文件；indent=2 用缩进提高可读性。
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @model_validator(mode="after")
    def resolve_auto_device(self) -> Self:
        # auto 的选择顺序：可用 CUDA → 可用 Apple MPS → CPU。
        # model_copy(update=...) 复制并替换部分字段，本 API 不会重新校验替换值。
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
            # 注意读变量流：下方给 update_dict 设置 pin_memory，但返回时用的是 train_update。
            # 所以当前实现实际只自动替换 device，并未通过此处开启 pin_memory。
            train_update = {"device": device}
            if device == "cuda":
                update_dict["pin_memory"] = True

            return self.model_copy(
                update={"train": self.train.model_copy(update=train_update)}
            )
        return self

    def _set_by_dotted_key(self, d: dict[str, Any], key: str, value: Any) -> None:
        # 输入 "train.batch_size"、32，将嵌套字典中的 d["train"]["batch_size"] 改为 32。
        # split(".") 拆层级；setdefault 在中间层不存在时先放入一个空字典。
        parts = key.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def with_overrides(self, overrides: list[str]) -> Self:
        # 输入如 ["train.batch_size=32"]，输出新配置；原配置字段不直接修改。
        # 为什么再校验：命令行改动也必须满足各字段的类型和范围要求。
        if not overrides:
            return self
        updates = self.model_dump()

        for ov in overrides:
            k, v = ov.split("=", 1)  # 最多拆一次，右边的值即使含 = 也能保留。
            # safe_load 使 "32" 变整数、"false" 变布尔值，而不是一直保留成字符串。
            self._set_by_dotted_key(updates, k, yaml.safe_load(v))
        return self.model_validate(updates)


# ======================================================================
# 3. 实时应用配置：模型路径、唤醒词、识别阈值和录音窗口
# ======================================================================
# 这里管理应用行为；训练网络和频谱的设置仍从已保存实验的 Config 中读取。
class InferConfig(BaseModel, frozen=True):
    model: str
    label_file: str
    device: str
    wakeword: str = "marvin"
    stopword: str = "stop"
    threshold: float = 0.7  # 最高类别概率超过这个值，才可能触发动作。
    cooldown_time: float = 1.0
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 16000  # 保留多少个最新采样点；16000 Hz 时，16000 点约为一秒。
    # default_factory 每次创建配置时新建一个字典，避免默认映射被不同实例共用。
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


# 学习练习（在单独脚本中导入相关配置类）：
# 预测：TrainConfig(batch_size=0) 能否创建？验证：调用它并查看 Pydantic 的字段错误信息。
# 修改：从有效 YAML 读取 cfg，调用 cfg.with_overrides(["train.batch_size=8"])，
# 比较新旧配置的 batch_size，确认原对象字段没有被直接改掉。
# 自己写：先声明只有 path 和 batch_size 的配置类，再补“从 YAML 读取”和“命令行覆盖”。
