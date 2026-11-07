"""M1 · 随机种子

目标
----
写一个 `seed_everything`，让“同一个 seed → 同一串随机数”在本项目用到的所有随机源上成立。

先读文档
--------
- PyTorch Reproducibility：https://docs.pytorch.org/docs/stable/notes/randomness.html
  （整页读完。里面讲了哪些东西 seed 管不住。）
- `random.seed`、`numpy.random.seed`、`torch.manual_seed`、`torch.Generator`

思考题
------
1. 这个项目里有几个独立的随机数来源？各在哪里被用到？（至少说出 3 个。）
2. 做个实验：只调用 `torch.manual_seed(0)`，然后打印 `np.random.rand()` 两次运行的结果。一样吗？
3. seed 固定了，GPU 上的训练结果就一定逐位相同吗？文档里举了哪些例外？
4. 为什么有些地方更推荐传一个独立的 `torch.Generator`，而不是依赖全局种子？
   （提示：想想“采样器的随机性”和“模型初始化的随机性”互相干扰的情况。）
"""

from __future__ import annotations

import torch


def seed_everything(seed: int) -> None:
    """给本项目用到的每个随机源设置种子。

    写完后自己验证：调用两次 seed_everything(1)，每次之后从每个随机源各取一个数，
    两次结果应该完全相同。
    """
    raise NotImplementedError("M1: seed_everything")


def make_generator(seed: int) -> torch.Generator:
    """返回一个用 seed 初始化好的、独立的 torch.Generator（CPU 上）。"""
    raise NotImplementedError("M1: make_generator")
