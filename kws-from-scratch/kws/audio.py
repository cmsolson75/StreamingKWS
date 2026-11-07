"""M2 · 音频 I/O 与基础操作

目标
----
把磁盘上的 WAV 变成统一格式的 tensor：**单声道、float32、目标采样率、固定长度**。
后面所有模块都假设输入已经是这个格式。

先读文档
--------
- soundfile.read：https://python-soundfile.readthedocs.io/ 重点：`dtype`、`always_2d`，返回的 shape 是 (帧数, 声道数) 还是反过来？
- torchaudio.functional.resample
- 分贝（dB）的定义：功率用 10·log10，幅度用 20·log10。为什么差 2 倍？

思考题
------
1. 16 位 PCM 的 WAV，用 `dtype="float32"` 读出来，数值范围是多少？用 `dtype="int16"` 呢？
2. 立体声转单声道，取平均和只取左声道有什么区别？
3. Speech Commands 里有的片段不到 1 秒。补零补在前面、后面、还是随机位置？
   训练时和评估时应该用同一种方式吗？
4. 一个振幅为 1 的正弦波，RMS 是多少？换算成 dB 是多少？（先手算，测试里会检查。）

写完的标准：`uv run pytest tests/test_m02_audio.py` 全绿。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

CropMode = Literal["left", "center", "random"]


def load_audio(path: str | Path, target_sr: int) -> torch.Tensor:
    """读一个音频文件，返回 shape 为 (T,) 的一维 float32 tensor。

    - 多声道时取各声道平均
    - 原始采样率和 target_sr 不同时重采样
    """
    raise NotImplementedError("M2: load_audio")


def pad_or_crop(
    wav: torch.Tensor,
    num_samples: int,
    mode: CropMode = "left",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """把最后一维变成恰好 num_samples 长。前面的维度原样保留。

    太长 → 截取一段；太短 → 补零。mode 决定位置：
    - "left"：原信号从第 0 个采样点开始（截取时取开头，补零时补在后面）
    - "center"：原信号居中（补零时多出来的那一个零放在后面）
    - "random"：随机位置，用 generator 产生随机数（generator 为 None 时用全局随机源）
    长度正好时原样返回。
    """
    raise NotImplementedError("M2: pad_or_crop")


def rms_db(wav: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """最后一维上的 RMS 能量，单位 dB（以 1.0 为参考）。返回去掉最后一维的 tensor。

    eps 防止 log(0)。思考：eps 应该加在 log 里面的哪个位置？
    """
    raise NotImplementedError("M2: rms_db")


def db_to_amplitude(db: float | torch.Tensor) -> float | torch.Tensor:
    """把一个**幅度**增益（dB）换算成乘法系数。例如 +6 dB 约等于乘 2。"""
    raise NotImplementedError("M2: db_to_amplitude")
