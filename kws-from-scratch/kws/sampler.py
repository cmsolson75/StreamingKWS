"""M4 · 按组加权采样

问题
----
四个数据集大小差别很大：UNKNOWN 有几万条，关键词每个只有几千条，SILENCE 是虚拟长度。
直接把它们拼起来随机抽，模型大部分时间都在学 UNKNOWN。
我们想要的是：**每次抽样先按配置的权重选一组，再在组内均匀抽一条**。

接口约定
--------
四个数据集会被 `ConcatDataset([kw, unk, sil, bg])` 拼起来。
采样器产出的是 ConcatDataset 的**全局下标**：第 g 组第 i 条 = 前 g 组的总长度 + i。

先读文档
--------
- torch.utils.data.Sampler：要实现 `__iter__` 和 `__len__`
- torch.multinomial、torch.randint 的 generator 参数
- torch.Generator.get_state / set_state

思考题
------
1. 和 `torch.utils.data.WeightedRandomSampler` 比，为什么这里不直接给每条样本一个权重？
   （算一下：要让 UNKNOWN 组总共占 20%，每条 UNKNOWN 的权重是多少？组的大小变了呢？）
2. 恢复训练时，为什么采样器也要能存档和读档？如果不存，恢复后模型会看到什么样的数据？
3. DataLoader 开了多个 worker 并且会预取（prefetch）。你在第 N 步存的采样器状态，
   真的对应“第 N 步之后的下一个样本”吗？（这是 M8 会遇到的坑，先记下来。）
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Sampler


class GroupWeightedSampler(Sampler[int]):
    """先按 weights 选组，再在组内均匀选一条，产出 num_samples 个全局下标。

    要求：
    - weights 不需要事先归一化
    - generator 为 None 时，自己新建一个（种子可以用 torch 全局随机源取）
    - 每产出一个下标就从 generator 取一次随机数（逐个生成，不要一开始就把整串都生成好）。
      这样 state_dict() 才能准确记录“下一次从哪里继续”
    - 同一个 generator 状态 → 同一串下标
    """

    def __init__(
        self,
        group_sizes: list[int],
        weights: list[float],
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError("M4: GroupWeightedSampler.__init__")

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError("M4: GroupWeightedSampler.__iter__")

    def __len__(self) -> int:
        raise NotImplementedError("M4: GroupWeightedSampler.__len__")

    def state_dict(self) -> dict[str, Any]:
        """返回能完整恢复随机序列的状态。"""
        raise NotImplementedError("M4: GroupWeightedSampler.state_dict")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError("M4: GroupWeightedSampler.load_state_dict")
