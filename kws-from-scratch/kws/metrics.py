"""M9 · 评估指标

为什么准确率不够
----------------
KWS 部署后，用户关心的是两件事：
- 说了关键词，它没反应（漏报，miss）
- 没说关键词，它乱触发（误报，false alarm），通常按“每小时多少次”算
一个 95% 准确率的模型，如果每小时误报 30 次，根本没法用。

先读文档
--------
- 混淆矩阵、精确率、召回率、F1 的定义（任何一本机器学习教材，或 scikit-learn 文档里的说明）
- torch.bincount（提示：混淆矩阵可以不写双重循环）

思考题
------
1. 混淆矩阵的行是真实标签还是预测标签？两种约定都有，你选了哪种，写在 docstring 里。
2. 某个类在测试集里一次都没出现，它的召回率是多少？某个类从没被预测过，它的精确率呢？
3. 验证集 1000 条、准确率 95%，这个数字的 95% 置信区间大约多宽？
   两个模型差 0.5%，能说明哪个更好吗？
4. 流式检测里，一次“命中”的判定需要时间容差。容差设太大或太小分别会怎样？
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    raise NotImplementedError("M9: accuracy")


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(C, C) 的 int64 矩阵，cm[i, j] = 真实为 i、预测为 j 的样本数。"""
    raise NotImplementedError("M9: confusion_matrix")


def precision_recall_f1(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从混淆矩阵算每个类的精确率、召回率、F1，各是 (C,) 的 float tensor。分母为 0 时该项记为 0。"""
    raise NotImplementedError("M9: precision_recall_f1")


@dataclass(frozen=True)
class DetectionResult:
    hits: int
    misses: int
    false_alarms: int


def match_detections(
    predicted: list[tuple[float, str]],
    reference: list[tuple[float, str]],
    tolerance_s: float,
) -> DetectionResult:
    """把流式检测输出的 (时间, 标签) 和标注的 (时间, 标签) 配对。

    - 标签相同且时间差 <= tolerance_s 才算命中
    - 每个标注最多被一个预测命中，每个预测最多命中一个标注
    - 没被命中的标注是漏报；没命中任何标注的预测是误报
    思考：两个预测都在同一个标注附近时，怎么配对？（贪心按时间顺序就够用，但要想清楚它什么时候会错。）
    """
    raise NotImplementedError("M9: match_detections")


def false_alarms_per_hour(false_alarms: int, audio_seconds: float) -> float:
    raise NotImplementedError("M9: false_alarms_per_hour")
