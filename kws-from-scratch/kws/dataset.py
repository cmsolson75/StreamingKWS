"""M4 · 数据集

目标
----
把 Speech Commands 组织成模型要的四类样本，每个样本返回 `(waveform, label_index)`：

| 数据集 | 来源 | 标签 |
|---|---|---|
| KeywordDataset | 你选的关键词目录 | 关键词本身 |
| UnknownDataset | 其余所有词的目录 | `<UNKNOWN>` |
| SilenceDataset | 现场合成的极小噪声 | `<SILENCE>` |
| BackgroundDataset | `_background_noise_` 里长录音的随机片段 | `<SILENCE>` |

先读文档
--------
- PyTorch Data 教程与 API：https://docs.pytorch.org/docs/stable/data.html
  重点：map-style Dataset 要实现哪两个方法；ConcatDataset 怎么把下标映射回各子数据集
- Speech Commands 论文第 7 节（数据切分）：https://arxiv.org/abs/1804.03209
- 数据集目录里的 README.md、validation_list.txt、testing_list.txt（M0 下载后自己打开看）

目录长这样（tests/conftest.py 里的假数据也是这个结构）：

    root/
      marvin/0a7c2a8d_nohash_0.wav
      ...
      _background_noise_/white_noise.wav
      validation_list.txt   # 每行一个相对路径，如 "marvin/0a7c2a8d_nohash_0.wav"
      testing_list.txt

思考题
------
1. 为什么官方切分按“说话人”（文件名 `_nohash_` 前面那段）分组，而不是随机切？随机切会让哪个指标虚高？
2. 为什么需要 `<UNKNOWN>` 和 `<SILENCE>` 两个类？如果只训练 12 个关键词，部署后会出什么问题？
3. SilenceDataset 和 BackgroundDataset 没有固定的“文件列表”，它们的 `__len__` 返回什么才合理？
4. 在 `__getitem__` 里用随机数（随机裁剪、随机噪声），开多个 DataLoader worker 时会不会每个 worker 产生一样的随机数？
   做实验验证，别只靠猜。
5. （扩展）想把音频缓存在内存里加速。num_workers=4 时，缓存会存几份？总共占多少内存？

写完的标准：`uv run pytest tests/test_m04_data.py` 全绿。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]
WaveTransform = Callable[[torch.Tensor], torch.Tensor]

UNKNOWN = "<UNKNOWN>"
SILENCE = "<SILENCE>"
BACKGROUND_DIR = "_background_noise_"


def build_labels(keywords: list[str]) -> list[str]:
    """标签表：关键词按给定顺序在前，然后是 UNKNOWN、SILENCE。

    思考：为什么标签顺序必须固定，并且要和模型一起保存？
    """
    raise NotImplementedError("M4: build_labels")


def list_split_files(root: str | Path, split: Split) -> list[Path]:
    """返回某个切分里的全部 .wav 文件（绝对路径，排好序）。

    - "val" / "test"：读 validation_list.txt / testing_list.txt
    - "train"：所有词目录下的 wav，去掉 val 和 test 里出现过的
    - 永远不包含 _background_noise_ 目录
    思考：为什么要排序？
    """
    raise NotImplementedError("M4: list_split_files")


class KeywordDataset(Dataset):
    """只包含 keywords 里那些词的样本。返回 (waveform (num_samples,), label_index)。

    transform 作用在定长波形上（M5 的数据增强从这里接入）。
    提示：KeywordDataset 和 UnknownDataset 有大量重复逻辑，可以先抽一个基类。
    """

    def __init__(
        self,
        root: str | Path,
        split: Split,
        keywords: list[str],
        labels: list[str],
        num_samples: int,
        sample_rate: int = 16000,
        transform: WaveTransform | None = None,
    ) -> None:
        raise NotImplementedError("M4: KeywordDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        raise NotImplementedError


class UnknownDataset(Dataset):
    """不在 keywords 里的所有词，标签统一是 UNKNOWN。参数含义同 KeywordDataset。"""

    def __init__(
        self,
        root: str | Path,
        split: Split,
        keywords: list[str],
        labels: list[str],
        num_samples: int,
        sample_rate: int = 16000,
        transform: WaveTransform | None = None,
    ) -> None:
        raise NotImplementedError("M4: UnknownDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        raise NotImplementedError


class SilenceDataset(Dataset):
    """合成的“几乎无声”样本：均匀或高斯的微弱噪声，RMS 不高于 noise_floor_db。标签 SILENCE。

    length 是“虚拟长度”。思考：同一个 idx 每次取到的应该是同一段音频吗？
    """

    def __init__(
        self,
        labels: list[str],
        num_samples: int,
        length: int = 1000,
        noise_floor_db: float = -60.0,
    ) -> None:
        raise NotImplementedError("M4: SilenceDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        raise NotImplementedError


class BackgroundDataset(Dataset):
    """从 root/_background_noise_ 下的长录音里随机裁 num_samples 长的片段。标签 SILENCE。

    提示：这些文件不大，在 __init__ 里一次性读进内存就行。
    """

    def __init__(
        self,
        root: str | Path,
        labels: list[str],
        num_samples: int,
        sample_rate: int = 16000,
        length: int = 1000,
    ) -> None:
        raise NotImplementedError("M4: BackgroundDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        raise NotImplementedError
