"""M6 · 模型

规则：**只看论文写 TC-ResNet8**，不要看 StreamingKWS 的 model.py。
论文：Temporal Convolution for Real-time Keyword Spotting on Mobile Devices
https://arxiv.org/abs/1904.03814（重点看 Figure 2 和 Table 1）

输入输出约定
------------
所有模型：输入 (B, n_mels, n_frames) 的 log-Mel 特征，输出 (B, num_classes) 的 logits（未经 softmax）。
n_frames 不固定：模型必须能处理任意长度（流式推理时会用到）。

先做 SimpleCNN 跑通全流程，再写 TC-ResNet8。

先读文档
--------
- nn.Conv1d / nn.Conv2d 的 shape 约定（哪一维是通道？）
- nn.BatchNorm1d、nn.Dropout、nn.AdaptiveAvgPool1d
- nn.Module 教程：https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

思考题
------
1. TC-ResNet 的核心思路：把 n_mels 当成**通道**，沿时间轴做 1D 卷积。
   和“把频谱当图片做 2D 卷积”相比，第一层的感受野、计算量各有什么不同？
2. 残差块里 stride=2 时，输入和输出的 shape 对不上，shortcut 要怎么处理？
3. Conv 后面紧跟 BatchNorm 时，Conv 的 bias 还有用吗？
4. 手算 TC-ResNet8（n_mels=40，width_mult=1）的参数量，和论文 Table 里的数字对照。
5. 模型最后为什么输出 logits 而不是概率？（提示：看 nn.CrossEntropyLoss 的文档。）

写完的标准：`uv run pytest tests/test_m06_model.py` 全绿。
"""

from __future__ import annotations

import torch
from torch import nn

from kws.config import ModelConfig


class SimpleCNN(nn.Module):
    """一个最简单的基线：几层 2D 卷积 + 全局池化 + 全连接。结构自己设计，参数量控制在 20 万以内。

    写它的目的不是效果，而是先让整条训练流程跑通，之后用来和 TC-ResNet8 对比。
    """

    def __init__(self, n_mels: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("M6: SimpleCNN.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M6: SimpleCNN.forward")


class TCResBlock(nn.Module):
    """论文 Figure 2 里的残差块（stride=1 和 stride=2 两种）。输入输出都是 (B, C, T)。"""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, kernel_size: int = 9
    ) -> None:
        super().__init__()
        raise NotImplementedError("M6: TCResBlock.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M6: TCResBlock.forward")


class TCResNet8(nn.Module):
    """论文里的 TC-ResNet8。width_mult 对应论文里的宽度系数 k（通道数都乘以它）。"""

    def __init__(
        self, n_mels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        raise NotImplementedError("M6: TCResNet8.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M6: TCResNet8.forward")


def build_model(cfg: ModelConfig, n_mels: int, num_classes: int) -> nn.Module:
    """按 cfg.name 构造模型。名字不认识时报 ValueError。"transformer" 等到 M12 再接。"""
    raise NotImplementedError("M6: build_model")


def count_parameters(model: nn.Module) -> int:
    """可训练参数的个数。"""
    raise NotImplementedError("M6: count_parameters")
