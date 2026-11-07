"""M3 · 特征：波形 → STFT → Mel → dB

目标
----
**不用 torch.stft、不用 torchaudio**，从零写出 log-Mel 频谱，并在数值上和 torchaudio 对齐。
写对之后，你就真正知道模型的输入是什么了。

整条流水线
----------
    波形 (B, T)
      → 分帧：(B, n_frames, n_fft)             frame_signal
      → 加窗 + 实数 FFT + 取模平方：(B, n_freqs, n_frames)   power_spectrogram
      → 乘 Mel 滤波器组：(B, n_mels, n_frames)   mel_filterbank
      → 转 dB：(B, n_mels, n_frames)             power_to_db

先读文档
--------
- torchaudio 特征提取教程：https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
- torch.stft 文档里 `center`、`pad_mode`、`window` 三个参数的说明（你要复现它的行为，但不能调用它）
- torch.fft.rfft、torch.hann_window（注意 `periodic` 参数）、Tensor.unfold、torch.nn.functional.pad 的 mode="reflect"
- Mel 刻度（HTK 公式）：自己查维基百科 “Mel scale”

思考题
------
1. n_fft=400、hop_length=160、采样率 16 kHz：窗长和帧移各是多少毫秒？1 秒音频有多少帧（center=True 时）？
2. rfft 输出多少个频点？为什么不是 n_fft 个？每个频点间隔多少 Hz？
3. 为什么分帧前要在两端各补 n_fft // 2 个点？“reflect” 补法和补零有什么区别？
4. Mel 滤波器是三角形的。第 m 个三角形的左、顶点、右三个频率从哪里来？
   低频的三角形窄、高频的宽，为什么这样设计？
5. 模型为什么吃 dB 而不是功率本身？（看一眼两者的数值范围。）
6. LogMelSpectrogram 里的窗函数和滤波器组为什么要用 register_buffer，而不是普通属性或 nn.Parameter？

写完的标准：`uv run pytest tests/test_m03_features.py` 全绿。
"""

from __future__ import annotations

import torch
from torch import nn


def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    """HTK 公式：Hz → Mel。"""
    raise NotImplementedError("M3: hz_to_mel")


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """hz_to_mel 的反函数。"""
    raise NotImplementedError("M3: mel_to_hz")


def mel_filterbank(
    n_freqs: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """返回 shape (n_freqs, n_mels) 的三角滤波器矩阵。

    要和 `torchaudio.functional.melscale_fbanks(..., norm=None, mel_scale="htk")` 一致。
    f_max 为 None 时取奈奎斯特频率。
    步骤：在 Mel 刻度上等距取 n_mels + 2 个点 → 换回 Hz → 对每个 FFT 频点算它在每个三角形上的高度。
    """
    raise NotImplementedError("M3: mel_filterbank")


def frame_signal(
    wav: torch.Tensor, n_fft: int, hop_length: int, center: bool = True
) -> torch.Tensor:
    """(..., T) → (..., n_frames, n_fft)。center=True 时先在两端做 reflect 补齐。"""
    raise NotImplementedError("M3: frame_signal")


def power_spectrogram(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """(..., T) → (..., n_freqs, n_frames) 的功率谱 |X|²。

    window 为 None 时用长度 n_fft 的 periodic Hann 窗。
    要和 `torch.stft(..., center=True, pad_mode="reflect", return_complex=True).abs() ** 2` 一致。
    """
    raise NotImplementedError("M3: power_spectrogram")


def power_to_db(
    power: torch.Tensor, amin: float = 1e-10, top_db: float | None = None
) -> torch.Tensor:
    """功率 → dB（参考值 1.0）。

    - 小于 amin 的值先夹到 amin
    - top_db 不为 None 时，再把每个样本（最后两维）里低于“该样本最大值 - top_db”的部分夹上去
    思考：top_db 防的是什么问题？
    """
    raise NotImplementedError("M3: power_to_db")


class LogMelSpectrogram(nn.Module):
    """(B, T) 波形 → (B, n_mels, n_frames) 的 log-Mel 特征。

    要求：
    - __init__ 里算好窗函数和滤波器组，用 register_buffer 注册
      （这样 model.to(device) 会一起搬过去，state_dict 里也有它们）
    - forward 只做计算，不创建新的常量 tensor
    """

    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int) -> None:
        super().__init__()
        raise NotImplementedError("M3: LogMelSpectrogram.__init__")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M3: LogMelSpectrogram.forward")


class SpecAugment(nn.Module):
    """在 (B, n_mels, n_frames) 特征上做频率遮挡和时间遮挡（SpecAugment 论文里的简化版）。

    - 只在 self.training 为 True 时生效；eval 模式下原样返回
    - freq_mask / time_mask 是遮挡宽度的上限，宽度在 [0, 上限] 里随机取；为 0 表示关闭
    - 每个样本各自随机，各遮一条频带和一段时间
    思考：被遮的位置填什么值？填 0 对 dB 特征意味着什么？（自己决定，写进学习日志。）
    论文：https://arxiv.org/abs/1904.08779
    """

    def __init__(self, freq_mask: int, time_mask: int) -> None:
        super().__init__()
        raise NotImplementedError("M3: SpecAugment.__init__")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M3: SpecAugment.forward")
