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


import soundfile as sf
from torchaudio.functional import resample

import torch

CropMode = Literal["left", "center", "random"]

## 这个函数用来读取dataset里的音频，然后将他们转成neural network可以处理的tensor格式
def load_audio(path: str | Path, target_sr: int) -> torch.Tensor:
    """读一个音频文件，返回 shape 为 (T,) 的一维 float32 tensor。

    - 多声道时取各声道平均
    - 原始采样率和 target_sr 不同时重采样
    """
    data, samplerate = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)  # 多声道取平均
    wav = torch.from_numpy(data)  # 音频信号要被转成 tensor 才让 neural network 处理
    if samplerate != target_sr:
        wav = resample(wav, orig_freq=samplerate, new_freq=target_sr)## 这步是把音频从orig_frequency重采样到new_frequency，返回的wav是tensor格式

    return wav ## 这里是把音频信号转成 tensor让neural network处理

    raise NotImplementedError("M2: load_audio")


## 这个函数用来把音频信号的长度裁剪成恰好 num_samples 长，用来给 neural network 处理。
def pad_or_crop(
    wav: torch.Tensor, ## 指这里wav期待的数据格式是tensor
    num_samples: int,
    mode: CropMode = "left",
    generator: torch.Generator | None = None,
) -> torch.Tensor: ## -> torch.Tensor 表示这个函数一定要返回一个tensor
    """把最后一维变成恰好 num_samples 长。前面的维度原样保留。

    太长 → 截取一段；太短 → 补零。mode 决定位置：
    - "left"：原信号从第 0 个采样点开始（截取时取开头，补零时补在后面）
    - "center"：原信号居中（补零时多出来的那一个零放在后面）
    - "random"：随机位置，用 generator 产生随机数（generator 为 None 时用全局随机源）
    长度正好时原样返回。
    """
    length = wav.shape[-1] ## shape是Tensor的形状，[-1]表示取最后一维
    if mode == "left":
        if length == num_samples: ## 长度正好直接返回
            return wav
        
        if length > num_samples:
            wav = wav[..., :num_samples] # 太长时，用切片截取前面的一段，syntax是这样的：数组[开始位置:结束位置]

        if length < num_samples: ## 太短时，补零
            padding = num_samples - length ## 计算需要补零的长度
            wav = torch.nn.functional.pad(wav, (0, padding)) ## (0, padding) 的意思是：左边补 0 个，右边补 padding 个也就是上面计算好的补零的长度

    if mode == 'center':
        if length == num_samples:
            return wav

        if length > num_samples:
            start = (length - num_samples) // 2 ## 计算截取的起始位置
            wav = wav[..., start:start + num_samples] ## 截取中间的一段

        if length < num_samples:
            padding = num_samples - length
            wav = torch.nn.functional.pad(wav, (padding // 2, padding - padding //2)) ## 这里是把补零的长度平均分配到前后两边

    if mode == 'random':
        if length == num_samples:
            return wav

        if length > num_samples:
            start = torch.randint(0, length - num_samples + 1, 1, generator=generator).item() 
            ## 这里是随机生成一个起始位置
            ## torch.randint(low, high, size, generator=...)
            ## 它生成随机整数，范围是：low ≤ 随机整数 < high
            wav = wav[..., start:start + num_samples] ## 截取随机位置的一段
    
    raise NotImplementedError("M2: pad_or_crop")


def rms_db(wav: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """最后一维上的 RMS 能量，单位 dB（以 1.0 为参考）。返回去掉最后一维的 tensor。

    eps 防止 log(0)。思考：eps 应该加在 log 里面的哪个位置？
    """
    rms = torch.sqrt(torch.mean(wav ** 2, dim=-1)) 
    ## 计算最后一维的均方根
    db = 20 * torch.log10(rms + eps)
    return db
    ## 这里把RMS换算成db，公式是 20 * log10（RMS + eps），eps是为了防止log(0)的情况才加的一个小数值，防止出现负无穷的情况

    raise NotImplementedError("M2: rms_db")


def db_to_amplitude(db: float | torch.Tensor) -> float | torch.Tensor:
    """把一个**幅度**增益（dB）换算成乘法系数。例如 +6 dB 约等于乘 2。
    """
    amplitude = 10 ** (db / 20) ## 这里是把db换算成幅度增益的公式，公式是 10 ** (db / 20)，因为幅度增益的定义是 20 * log10(amplitude)，所以要反过来就是 amplitude = 10 ** (db / 20)
    return amplitude

    raise NotImplementedError("M2: db_to_amplitude")
