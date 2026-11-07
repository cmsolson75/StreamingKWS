"""M7 · 手写优化器和梯度裁剪

目标
----
继承 torch.optim.Optimizer，自己写 SGD（带 momentum）和 AdamW，
在同样的初始权重、同样的梯度下，和 PyTorch 官方实现逐步一致。
训练系统里用你自己的 AdamW。

先读文档
--------
- torch.optim.Optimizer 的源码结构：param_groups、state、defaults 分别是什么
  （读 https://docs.pytorch.org/docs/stable/optim.html 和 SGD / AdamW 页面上的伪代码）
- Adam 论文 Algorithm 1：https://arxiv.org/abs/1412.6980
- AdamW 论文（Decoupled Weight Decay）：https://arxiv.org/abs/1711.05101
- torch.no_grad，以及 Tensor 的原地操作（mul_、add_、addcmul_、addcdiv_）

思考题
------
1. Adam 的偏差修正（bias correction）修的是什么偏差？前几步不修会怎样？
2. Adam + L2 正则 和 AdamW 的区别，用公式写出来。哪一个的 weight decay 会被自适应学习率“缩放”？
3. 为什么 step() 要用 @torch.no_grad() 装饰？
4. 优化器的 state 存在哪里？为什么恢复训练时必须连优化器一起存？（M8 会考。）
5. 梯度裁剪的“总范数”是把所有参数的梯度拼成一个大向量后的 L2 范数。
   为什么按总范数缩放，而不是每个参数各自裁？

写完的标准：`uv run pytest tests/test_m07_training.py -k "sgd or adamw or clip"` 全绿。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    """带 momentum 的 SGD，和 torch.optim.SGD(lr, momentum, weight_decay) 在 dampening=0、nesterov=False 时一致。"""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        raise NotImplementedError("M7: SGD.__init__")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        raise NotImplementedError("M7: SGD.step")


class AdamW(Optimizer):
    """和 torch.optim.AdamW(lr, betas, eps, weight_decay) 一致（amsgrad=False）。"""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        raise NotImplementedError("M7: AdamW.__init__")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        raise NotImplementedError("M7: AdamW.step")


@torch.no_grad()
def clip_grad_norm(
    params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6
) -> torch.Tensor:
    """总范数超过 max_norm 时，把所有梯度按同一比例缩小。返回**裁剪前**的总范数。
    要和 torch.nn.utils.clip_grad_norm_ 一致。没有梯度的参数跳过。
    """
    raise NotImplementedError("M7: clip_grad_norm")
