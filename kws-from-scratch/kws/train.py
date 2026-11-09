"""M7 · 训练循环与启动入口

运行方式
--------
    uv run python -m kws.train --config configs/config.yaml train.max_steps=200 train.num_workers=0

**写的顺序很重要**（每一步验证通过再往下）：
1. train_step + 一个最简单的循环，不加任何技巧
2. 过拟合一个 batch：关掉 dropout 和增强，loss 要接近 0。做不到就是有 bug，先别往下走
3. 加 evaluate，定期在验证集上评估
4. 逐个加入：scheduler → 梯度裁剪 → EMA → AMP。每加一个都和之前的曲线比较
5. 加存档和恢复（M8）

先读文档
--------
- argparse：parse_known_args（为什么不用 parse_args？）
- Module.train() / Module.eval() 会影响哪些层？
- torch.inference_mode 和 torch.no_grad 的区别
- AMP：https://docs.pytorch.org/docs/stable/notes/amp_examples.html（torch.autocast、torch.amp.GradScaler）
- A Recipe for Training Neural Networks：https://karpathy.github.io/2019/04/25/recipe/（写之前完整读一遍）

思考题
------
1. optimizer.zero_grad() 放在 backward 之前还是 step 之后？set_to_none=True 有什么好处？
2. 用了 GradScaler 时，梯度裁剪必须在哪一步之后做？为什么？
3. evaluate 时如果忘了 model.eval()，BatchNorm 和 Dropout 分别会出什么问题？
4. 训练用的是无限采样器，那 “epoch” 这个概念在这里还有意义吗？用 step 计数有什么好处？
5. 特征提取（LogMelSpectrogram）放在 Dataset 里（CPU worker 上）做，还是放在训练循环里（GPU 上）做？各有什么取舍？
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from kws.config import Config

import argparse


def parse_args(argv: Sequence[str] | None = None) -> tuple[str, list[str]]:
    """解析命令行：必须有 --config <path>，其余形如 a.b=c 的参数作为覆盖列表返回。

    例：["--config", "c.yaml", "train.lr=0.01"] → ("c.yaml", ["train.lr=0.01"])
    """
    ## 第一步：创建一个解析命令行的工具
    parser = argparse.ArgumentParser()

    ## 第二步：告诉这个工具规则
    ## 意思是：1.识别 --config，并取它后面的一个值
    parser.add_argument("--config", required=True) ## 这里是保证接收到的参数必须有config这个参数，required=True表示必须有这个参数

    ## 第三步：提取认识的参数，留下其余参数
    ## args 是一个通过属性取值的对象；overrides 是剩余的字符串列表
    arv, overrides = parser.parse_known_args(
        ["--config", "c.yamal", "train.lr=0.01"]
    )



    raise NotImplementedError("M7: parse_args")


def train_step(
    model: nn.Module,
    features: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    clip_norm: float | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """一步训练：搬到设备 → 特征 → 前向 → loss → 反向 →（裁剪）→ 更新。

    返回 {"loss": float, "acc": float, "grad_norm": float}。
    不在这里调用 scheduler.step() 和 ema.update()，它们由外层循环负责。思考：为什么这样分工？
    """
    raise NotImplementedError("M7: train_step")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    features: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """在整个 loader 上算平均 loss 和准确率，返回 {"loss": float, "acc": float}。

    要求：
    - 按样本数加权平均（最后一个 batch 可能比较小）
    - 结束时把 model 恢复成调用前的 train/eval 状态
    """
    raise NotImplementedError("M7: evaluate")


def train(cfg: Config) -> None:
    """完整训练流程。没有单元测试，验收标准是：在真实数据上跑出 > 90% 的验证准确率，
    并且 run 目录里有 config.json、provenance.json、metrics.jsonl 和 checkpoints/。

    建议的骨架（自己决定每一步怎么写）：
        设备 → seed → run 目录 → 标签 → DataLoader → 特征模块 → 模型 → 优化器 → scheduler → EMA
        →（M8：如果 cfg.env.resume 不为空，就恢复）
        → for step in range(start_step, max_steps): train_step → scheduler → EMA → 记日志 → 定期评估 + 存档
        → 最后用最佳 EMA 模型在测试集上评估
    """
    raise NotImplementedError("M7: train")


def main(argv: Sequence[str] | None = None) -> None:
    raise NotImplementedError("M7: main")


if __name__ == "__main__":
    main()
