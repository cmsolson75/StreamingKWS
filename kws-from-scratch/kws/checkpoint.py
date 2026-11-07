"""M8 · 存档与严格恢复

目标
----
训练到第 100 步存档 → 关掉进程 → 新进程读档 → 再训练 100 步，
结果要和“一口气训练 200 步”**完全一样**（CPU、num_workers=0 的条件下）。

这一关通过，说明你真正知道“训练状态”包含哪些东西。漏掉任何一个，测试都会失败。

先读文档
--------
- Saving and Loading Models：https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- torch.load 的 weights_only 参数：为什么默认是 True？它防的是什么攻击？
- random.getstate / setstate、numpy.random.get_state / set_state、
  torch.get_rng_state / set_rng_state、torch.cuda.get_rng_state_all
- os.replace（原子替换）

思考题
------
1. 列一张清单：恢复训练需要哪些状态？（至少 7 项。写完测试通过后，回来对照你漏了哪些。）
2. numpy 的 RNG 状态里有 numpy 数组。torch.load(weights_only=True) 能读回 numpy 数组吗？
   不能的话，存之前要把它转成什么？
3. 存档写到一半，进程被 kill 了。下次启动读到一个损坏的文件。怎样写才能避免？
4. latest.json / best.json 这种“指针文件”有什么用？为什么不直接用文件名找最新的？
5. 存档里的 step 表示“已经完成的步数”还是“下一步要跑的步数”？两种约定都行，但你必须选一个并在恢复时用对。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def capture_rng_state() -> dict[str, Any]:
    """收集 Python random、NumPy、torch CPU（以及可用时 CUDA）的随机状态。

    要求：返回值存进 torch.save 后，能用 torch.load(..., weights_only=True) 读回来。
    """
    raise NotImplementedError("M8: capture_rng_state")


def restore_rng_state(state: dict[str, Any]) -> None:
    """capture_rng_state 的逆操作。"""
    raise NotImplementedError("M8: restore_rng_state")


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    ema: Any | None = None,
    scaler: Any | None = None,
    sampler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """把训练状态写到 path。

    - 凡是传进来的、有 state_dict() 的对象都要存
    - 随机状态也要存（capture_rng_state）
    - 原子写入：先写临时文件，再替换
    - 存出来的文件必须能用 torch.load(path, weights_only=True) 读取
    """
    raise NotImplementedError("M8: save_checkpoint")


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    ema: Any | None = None,
    scaler: Any | None = None,
    sampler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """读回状态，原地恢复到传入的对象里。返回 {"step": int, "extra": dict}。

    只做评估时 optimizer 等可以不传（为 None 就跳过）。
    用 weights_only=True 读取。
    """
    raise NotImplementedError("M8: load_checkpoint")


def write_pointer(
    checkpoint_dir: str | Path, name: str, checkpoint_path: str | Path, step: int
) -> None:
    """写 checkpoint_dir/<name>.json，例如 latest.json、best.json。

    内容至少包括存档文件名（相对 checkpoint_dir）和 step。同样要原子写入。
    """
    raise NotImplementedError("M8: write_pointer")


def read_pointer(checkpoint_dir: str | Path, name: str) -> Path | None:
    """读指针，返回存档的绝对路径；指针不存在时返回 None。"""
    raise NotImplementedError("M8: read_pointer")
