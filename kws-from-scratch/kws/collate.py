"""M4 · collate：把一组样本拼成一个 batch

背景
----
DataLoader 每次从 Dataset 拿到 batch_size 个 `(waveform, label)`，然后调用 collate_fn 把它们拼成 batch。
默认的 collate 用 torch.stack，要求所有 waveform 一样长。
本项目在 Dataset 里已经把音频统一成 1 秒，所以默认 collate 能用。
但真实任务里经常是变长的（整句话、不同长度的录音），这时你必须自己写 collate。

这个练习**故意**处理变长输入，让你学到 collate 真正要解决的问题：补齐 + 记住哪里是补的。

先读文档
--------
- DataLoader 的 collate_fn 参数：https://docs.pytorch.org/docs/stable/data.html#dataloader-collate-fn
- torch.nn.utils.rnn.pad_sequence（可以用来对照，但先自己写一遍）
- typing.NamedTuple

思考题
------
1. 默认 collate 遇到长度不同的 tensor 会报什么错？自己构造一下看看。
2. mask 里 True 表示“真实数据”还是“补出来的”？两种约定都有人用，PyTorch 的
   `nn.MultiheadAttention` 的 key_padding_mask 用的是哪一种？（M12 会用到。）
3. 补齐后再算 Mel 频谱，补零部分会产生帧。你怎么从“采样点的长度”换算出“帧的长度”？
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class Batch(NamedTuple):
    waveforms: torch.Tensor  # (B, T_max) float32，补零
    lengths: torch.Tensor  # (B,) int64，每条的真实长度
    labels: torch.Tensor  # (B,) int64
    mask: torch.Tensor  # (B, T_max) bool，True = 真实数据


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """(B,) 长度 → (B, max_len) 的 bool mask，True 表示有效位置。max_len 为 None 时取最大长度。

    提示：不用写循环。想想 torch.arange 和广播。
    """
    raise NotImplementedError("M4: lengths_to_mask")


def pad_collate(samples: list[tuple[torch.Tensor, int]]) -> Batch:
    """把 [(waveform (T_i,), label), ...] 拼成 Batch。不能修改输入的 tensor。"""
    raise NotImplementedError("M4: pad_collate")
