"""M7 · 学习率计划：线性 warmup + cosine 衰减

形状（先在纸上画出来）
---------------------
    倍率
    1.0 |      ____
        |     /    `--._
        |    /          `-.
        |   /              `.
    min |__/_________________`.___
        0  warmup            max_steps   step

- warmup 阶段：从 1/warmup_steps 线性升到 1.0（第 0 步是 1/warmup_steps，第 warmup_steps-1 步是 1.0）
- 之后：半个余弦周期，从 1.0 降到 min_ratio，在 max_steps 处到达 min_ratio
- 超过 max_steps：保持 min_ratio
- warmup_steps 为 0 时没有 warmup

先读文档
--------
- torch.optim.lr_scheduler.LambdaLR：lr_lambda 收到的参数是什么？scheduler 创建时会不会先调用一次？
- SGDR 论文里的 cosine 公式：https://arxiv.org/abs/1608.03983

思考题
------
1. 为什么大 batch、Adam 类优化器的训练开头需要 warmup？
2. 训练循环里 optimizer.step() 和 scheduler.step() 的先后顺序是什么？反过来会怎样？（PyTorch 会给你一个警告。）
3. 恢复训练时如果没恢复 scheduler，学习率会怎样变化？
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(step: int, warmup_steps: int, max_steps: int, min_ratio: float = 0.0) -> float:
    """返回第 step 步的学习率倍率（乘在基础学习率上）。"""
    raise NotImplementedError("M7: warmup_cosine")


def build_scheduler(
    optimizer: Optimizer, warmup_steps: int, max_steps: int, min_ratio: float = 0.0
) -> LambdaLR:
    """用 warmup_cosine 构造 LambdaLR。

    提示：lambda 会捕获外部变量（闭包）。但 LambdaLR 的 state_dict 存不下 lambda 本身，
    思考这对“恢复训练”有没有影响。
    """
    raise NotImplementedError("M7: build_scheduler")
