"""M10 · 流式检测：滑动窗口 → 概率平滑 → 状态机

整体流程
--------
    连续音频 ──每 hop_s 秒──▶ 取最近 window_s 秒 ──▶ 特征 + 模型 ──▶ softmax 概率
            ──▶ PosteriorSmoother（对最近几次的概率取平均）
            ──▶ KeywordStateMachine（阈值 + 冷却 + 状态切换）──▶ 事件

应用逻辑（和 StreamingKWS 相同）
------------------------------
- IDLE：等唤醒词（marvin）。听到 → 进入 LISTENING，发出 "wake" 事件
- LISTENING：每听到一个数字词，发出 "digit" 事件并记下数字；
  听到停止词（stop）→ 发出 "stop" 事件（带上收集到的全部数字），回到 IDLE
- IDLE 状态下的数字和停止词忽略

**关键：先写这个离线版本**（输入一整段波形，输出事件列表），用测试验证；
确认正确后再在 app.py 里接麦克风。直接对着麦克风调试，你永远不知道是模型错、阈值错还是代码错。

先读文档
--------
- enum.Enum、dataclasses.dataclass(frozen=True)
- collections.deque(maxlen=...)
- 论文：Streaming Keyword Spotting on Mobile Devices https://arxiv.org/abs/2005.06720

思考题
------
1. 同一个词持续约 1 秒，而你每 0.1 秒推理一次，它会被识别很多次。冷却（cooldown）怎么防止重复触发？
2. <UNKNOWN> 和 <SILENCE> 的概率超过阈值时，应该触发冷却吗？
   （StreamingKWS 原代码在这里有个 bug：背景声持续“触发”，把真正的关键词挡掉了。你的实现不能有这个问题。）
3. 平滑窗口越大，误报越少，但延迟越高。延迟大约增加多少？
4. 窗口长度必须和训练时的片段长度一致吗？不一致会怎样？
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from torch import nn


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"


@dataclass(frozen=True)
class Event:
    kind: str  # "wake" | "digit" | "stop"
    label: str
    time: float  # 秒，触发时刻（= 当前窗口结束的时间）
    digits: tuple[int, ...] = field(default_factory=tuple)  # 只有 "stop" 事件填写


class PosteriorSmoother:
    """对最近 window 次的概率向量取平均。window=1 等于不平滑。reset() 清空历史。"""

    def __init__(self, window: int) -> None:
        raise NotImplementedError("M10: PosteriorSmoother.__init__")

    def update(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("M10: PosteriorSmoother.update")

    def reset(self) -> None:
        raise NotImplementedError("M10: PosteriorSmoother.reset")


class KeywordStateMachine:
    """每次收到一个（平滑后的）概率向量和当前时间，返回本次产生的事件列表（通常为空）。

    触发规则：
    - 取概率最高的标签；概率 < threshold 时什么都不做
    - <UNKNOWN> / <SILENCE> 永远不触发，也不影响冷却
    - 一个“有意义的”词触发后，cooldown_s 秒内不再触发任何词
    - 有意义的词：IDLE 下只有 wakeword；LISTENING 下是 digit_map 里的词和 stopword
    """

    def __init__(
        self,
        labels: Sequence[str],
        wakeword: str,
        stopword: str,
        digit_map: Mapping[str, int],
        threshold: float,
        cooldown_s: float,
    ) -> None:
        raise NotImplementedError("M10: KeywordStateMachine.__init__")

    @property
    def state(self) -> State:
        raise NotImplementedError

    def step(self, probs: np.ndarray, t: float) -> list[Event]:
        raise NotImplementedError("M10: KeywordStateMachine.step")


class StreamingDetector:
    """离线模拟流式推理。

    run_on_waveform(wav)：把 wav 按每块 hop 个采样点依次写进一个 CircularBuffer(window 个采样点)，
    每写一块（并且缓冲区已经写满过一次之后）就：读缓冲区 → features → model → softmax → smoother → state_machine。
    返回全部事件。事件时间 = 已写入的采样点数 / sample_rate。

    要求：推理在 torch.inference_mode 下进行，model 处于 eval 模式。
    """

    def __init__(
        self,
        model: nn.Module,
        features: nn.Module,
        state_machine: KeywordStateMachine,
        smoother: PosteriorSmoother,
        sample_rate: int = 16000,
        window_s: float = 1.0,
        hop_s: float = 0.1,
    ) -> None:
        raise NotImplementedError("M10: StreamingDetector.__init__")

    def run_on_waveform(self, wav: np.ndarray | torch.Tensor) -> list[Event]:
        raise NotImplementedError("M10: StreamingDetector.run_on_waveform")
