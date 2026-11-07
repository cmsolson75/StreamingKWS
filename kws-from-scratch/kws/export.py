"""M11 · 导出 ONNX 并验证

目标
----
把训练好的模型导出成 ONNX，用 onnxruntime 推理，验证结果和 PyTorch 一致，再测延迟和文件大小。

先读文档
--------
- PyTorch ONNX：https://docs.pytorch.org/docs/stable/onnx.html
  注意新版本默认用 dynamo 导出器；dynamic_shapes / dynamic_axes 怎么声明“batch 维可变”
- onnxruntime.InferenceSession：https://onnxruntime.ai/docs/api/python/api_summary.html

思考题
------
1. 导出前为什么必须 model.eval()？不这样做，BatchNorm 和 Dropout 会被导出成什么？
2. 只导出模型，还是把 LogMelSpectrogram 也一起导出？各有什么好处和坑？
3. 结果“一致”应该用什么容差？为什么不能要求逐位相同？
4. （扩展）试试 torch.ao 的动态量化，模型小了多少？准确率掉了多少？
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def export_onnx(model: nn.Module, example_input: torch.Tensor, path: str | Path) -> Path:
    """把 model 导出到 path，batch 维（第 0 维）可变。返回 path。"""
    raise NotImplementedError("M11: export_onnx")


def max_abs_diff_vs_onnx(model: nn.Module, onnx_path: str | Path, x: torch.Tensor) -> float:
    """分别用 PyTorch 和 onnxruntime 推理同一个输入，返回输出的最大绝对差。"""
    raise NotImplementedError("M11: max_abs_diff_vs_onnx")
