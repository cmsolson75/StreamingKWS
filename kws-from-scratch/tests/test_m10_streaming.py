"""M10 验收测试：环形缓冲区、平滑、状态机、离线流式检测。不要修改。"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
from torch import nn

from kws.dataset import SILENCE, UNKNOWN
from kws.streaming.buffer import CircularBuffer
from kws.streaming.detector import (
    KeywordStateMachine,
    PosteriorSmoother,
    State,
    StreamingDetector,
)

LABELS = ["marvin", "stop", "one", "two", UNKNOWN, SILENCE]
DIGITS = {"one": 1, "two": 2}


# ---------------------------------------------------------------- buffer
def test_buffer_before_full_is_zero_padded():
    b = CircularBuffer(5)
    b.write(np.array([1, 2], dtype=np.float32))
    assert b.read().tolist() == [0, 0, 0, 1, 2]
    assert b.total_written == 2


def test_buffer_wraps_around():
    b = CircularBuffer(4)
    b.write(np.array([1, 2, 3], dtype=np.float32))
    b.write(np.array([4, 5, 6], dtype=np.float32))
    assert b.read().tolist() == [3, 4, 5, 6]


def test_buffer_chunk_longer_than_capacity():
    b = CircularBuffer(3)
    b.write(np.arange(10, dtype=np.float32))
    assert b.read().tolist() == [7, 8, 9]
    assert b.total_written == 10


def test_buffer_read_returns_copy():
    b = CircularBuffer(3)
    b.write(np.ones(3, dtype=np.float32))
    out = b.read()
    out[:] = 99
    assert b.read().tolist() == [1, 1, 1]


def test_buffer_concurrent_writes():
    b = CircularBuffer(1000)

    def writer():
        for _ in range(500):
            b.write(np.ones(7, dtype=np.float32))

    threads = [threading.Thread(target=writer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert b.total_written == 8 * 500 * 7
    assert (b.read() == 1).all()


# ---------------------------------------------------------------- smoother
def test_smoother_moving_average():
    s = PosteriorSmoother(window=2)
    assert np.allclose(s.update(np.array([1.0, 0.0])), [1.0, 0.0])
    assert np.allclose(s.update(np.array([0.0, 1.0])), [0.5, 0.5])
    assert np.allclose(s.update(np.array([0.0, 1.0])), [0.0, 1.0])
    s.reset()
    assert np.allclose(s.update(np.array([0.2, 0.8])), [0.2, 0.8])


# ---------------------------------------------------------------- state machine
def _onehot(label, p=0.95):
    v = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1))
    v[LABELS.index(label)] = p
    return v


def _sm(cooldown=1.0):
    return KeywordStateMachine(LABELS, "marvin", "stop", DIGITS, threshold=0.7, cooldown_s=cooldown)


def test_full_sequence():
    sm = _sm()
    events = []
    for t, word in [(0.0, "marvin"), (2.0, "one"), (4.0, "two"), (6.0, "stop")]:
        events += sm.step(_onehot(word), t)
    assert [(e.kind, e.label) for e in events] == [
        ("wake", "marvin"),
        ("digit", "one"),
        ("digit", "two"),
        ("stop", "stop"),
    ]
    assert events[-1].digits == (1, 2)
    assert sm.state is State.IDLE


def test_idle_ignores_digits_and_stop():
    sm = _sm()
    assert sm.step(_onehot("one"), 0.0) == []
    assert sm.step(_onehot("stop"), 2.0) == []
    assert sm.state is State.IDLE


def test_below_threshold_does_nothing():
    sm = _sm()
    assert sm.step(_onehot("marvin", p=0.5), 0.0) == []


def test_cooldown_prevents_repeats():
    sm = _sm(cooldown=1.0)
    events = []
    for i in range(10):  # 同一个词连续出现 0.9 秒
        events += sm.step(_onehot("marvin"), i * 0.1)
    assert len(events) == 1
    sm.step(_onehot("one"), 1.5)
    assert sm.step(_onehot("two"), 1.6) == []  # 冷却中
    assert sm.step(_onehot("two"), 2.6)[0].label == "two"


def test_silence_and_unknown_do_not_block_keywords():
    sm = _sm(cooldown=1.0)
    assert sm.step(_onehot(SILENCE), 0.0) == []
    assert sm.step(_onehot(UNKNOWN), 0.1) == []
    events = sm.step(_onehot("marvin"), 0.2)
    assert [e.kind for e in events] == ["wake"]


def test_new_session_clears_digits():
    sm = _sm()
    for t, w in [(0, "marvin"), (2, "one"), (4, "stop"), (6, "marvin"), (8, "two"), (10, "stop")]:
        events = sm.step(_onehot(w), float(t))
    assert events[-1].digits == (2,)


# ---------------------------------------------------------------- offline detector
SR = 1000  # 测试里用很低的采样率，跑得快


class CenterSampleModel(nn.Module):
    """假模型：看窗口正中间那个采样点的值决定类别。值 (k+1)/10 → 第 k 类；0 → SILENCE。"""

    def forward(self, x):
        center = x[:, x.shape[1] // 2]
        idx = torch.where(
            center > 0, torch.round(center * 10) - 1, torch.tensor(float(LABELS.index(SILENCE)))
        )
        return nn.functional.one_hot(idx.long(), len(LABELS)).float() * 20


def _segment(label, seconds):
    value = 0.0 if label is None else (LABELS.index(label) + 1) / 10
    return np.full(int(seconds * SR), value, dtype=np.float32)


def test_detector_on_synthetic_stream():
    plan = [None, "marvin", None, "one", None, "two", None, "stop", None]
    wav = np.concatenate([_segment(w, 0.8 if w else 0.6) for w in plan])
    det = StreamingDetector(
        CenterSampleModel(),
        nn.Identity(),
        _sm(cooldown=1.0),
        PosteriorSmoother(window=1),
        sample_rate=SR,
        window_s=1.0,
        hop_s=0.1,
    )
    events = det.run_on_waveform(wav)
    assert [e.label for e in events] == ["marvin", "one", "two", "stop"]
    assert events[-1].digits == (1, 2)
    # 事件时间 = 窗口结束时刻；窗口中心刚进入单词时触发，所以约晚 0.5 秒
    assert events[0].time == pytest.approx(0.6 + 0.5, abs=0.11)
