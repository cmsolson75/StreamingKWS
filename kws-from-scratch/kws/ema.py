"""M7 · 权重的指数移动平均（EMA）

想法
----
训练时另外保留一份“影子模型”，每步更新：
    shadow = decay · shadow + (1 - decay) · current
评估和部署用影子模型，它的权重更平滑，常常比当前模型稍好。

先读文档
--------
- copy.deepcopy、Module.parameters / buffers / named_parameters / named_buffers
- Module.requires_grad_、Module.eval
- 对照阅读（写完后再看）：ema_pytorch 库的实现，比较它多做了什么

思考题
------
1. decay=0.999 时，影子模型大约“记住”最近多少步？（推一下：权重衰减到 1/e 需要多少步。）
2. BatchNorm 的 running_mean / running_var 是 buffer 不是参数。它们应该做平均还是直接复制？
   num_batches_tracked 呢？（它是整数 tensor。）
3. 影子模型需要梯度吗？它应该处于 train 还是 eval 模式？
4. 恢复训练时，EMA 的影子权重需要存吗？只存当前模型、恢复时用当前模型初始化影子行不行？

写完的标准：`uv run pytest tests/test_m07_training.py -k ema` 全绿。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EMA:
    """用法：
        ema = EMA(model, decay=0.999)
        ...每个 optimizer.step() 之后...
        ema.update(model)
        ...评估时...
        evaluate(ema.model, ...)

    要求：
    - ema.model 是 model 的独立深拷贝，不需要梯度，处于 eval 模式
    - update：浮点参数按公式平均；buffer 直接复制
    - state_dict / load_state_dict 能完整保存和恢复（包括 decay 和更新次数）
    - copy_to(model)：把影子权重写回给另一个结构相同的模型
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        raise NotImplementedError("M7: EMA.__init__")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        raise NotImplementedError("M7: EMA.update")

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        raise NotImplementedError("M7: EMA.copy_to")

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError("M7: EMA.state_dict")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError("M7: EMA.load_state_dict")
