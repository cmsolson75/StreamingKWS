"""M5 · 波形数据增强

目标
----
让模型在训练时就见过“有噪声、音量忽大忽小”的语音，部署到真实麦克风上才不会崩。

两种增强
--------
- RandomGain：随机放大或缩小音量（单位 dB）
- RandomNoise：从噪声库里随机取一段，按随机的信噪比（SNR）混进去

SNR 的定义（先自己推导，再写代码）
---------------------------------
    SNR_dB = 10 · log10( P_signal / P_noise )，P 是平均功率（均方值）
要得到目标 SNR，噪声要乘以多大的系数 k？把 k 用 P_signal、P_noise、SNR_dB 表示出来，写进学习日志。

先读文档
--------
- 论文里的做法：Speech Commands 论文、以及任意一篇 “noise augmentation SNR” 的 KWS 论文
- torch.rand / torch.randint 的 generator 参数

思考题
------
1. 信号本身几乎是静音（功率接近 0）时，你的 SNR 公式会怎样？怎么防？
2. SNR = -5 dB 意味着噪声比语音还响。为什么配置里允许负数？
3. 混完噪声后幅度可能超过 1。需要处理吗？（想想 16 位 WAV 存不下的情况和模型训练的情况有什么不同。）
4. 下面的类都接受一个可选的 generator。如果你给训练用的每个 DataLoader worker 传**同一个** generator 对象，
   每个 worker 产生的随机增强会是什么样？所以默认值应该是什么？

写完的标准：`uv run pytest tests/test_m05_augment.py` 全绿，然后回到 kws/dataloaders.py 接上增强。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from kws.config import AugmentConfig

WaveTransform = Callable[[torch.Tensor], torch.Tensor]


def mix_at_snr(
    clean: torch.Tensor, noise: torch.Tensor, snr_db: float, eps: float = 1e-10
) -> torch.Tensor:
    """返回 clean + k * noise，k 使得混合后 clean 与 (k * noise) 的 SNR 等于 snr_db。

    clean 和 noise 形状相同（一维，长度 T）。
    """
    raise NotImplementedError("M5: mix_at_snr")


class RandomGain:
    """以概率 p 把波形乘上一个增益，增益（dB）在 [min_db, max_db] 均匀分布。"""

    def __init__(
        self, min_db: float, max_db: float, p: float, generator: torch.Generator | None = None
    ) -> None:
        raise NotImplementedError("M5: RandomGain.__init__")

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M5: RandomGain.__call__")


class RandomNoise:
    """以概率 p 从 noise_bank 随机选一段噪声，随机裁到和 wav 一样长，
    再按 [min_snr_db, max_snr_db] 里均匀分布的 SNR 混进去。

    noise_bank 里每段噪声都比 wav 长（背景噪声文件都是几十秒）。
    """

    def __init__(
        self,
        noise_bank: Sequence[torch.Tensor],
        min_snr_db: float,
        max_snr_db: float,
        p: float,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError("M5: RandomNoise.__init__")

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M5: RandomNoise.__call__")


class Compose:
    """按顺序依次应用。思考：先加噪再调增益，和先调增益再加噪，效果一样吗？"""

    def __init__(self, transforms: Sequence[WaveTransform]) -> None:
        raise NotImplementedError("M5: Compose.__init__")

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M5: Compose.__call__")


def load_noise_bank(noise_dir: str | Path, sample_rate: int) -> list[torch.Tensor]:
    """读取目录下全部 .wav，返回一维 tensor 列表。用 kws.audio.load_audio。"""
    raise NotImplementedError("M5: load_noise_bank")


def build_augment(cfg: AugmentConfig, noise_bank: Sequence[torch.Tensor]) -> WaveTransform | None:
    """cfg.enabled 为 False 时返回 None；否则按配置组装 Compose。"""
    raise NotImplementedError("M5: build_augment")
