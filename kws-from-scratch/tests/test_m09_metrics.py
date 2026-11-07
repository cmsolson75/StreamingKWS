"""M9 验收测试：评估指标。不要修改。"""

from __future__ import annotations

import pytest
import torch

from kws.metrics import (
    accuracy,
    confusion_matrix,
    false_alarms_per_hour,
    match_detections,
    precision_recall_f1,
)


def test_accuracy():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0], [0.0, 5.0]])
    assert accuracy(logits, torch.tensor([0, 1, 1, 1])) == pytest.approx(0.75)


def test_confusion_matrix_rows_are_targets():
    preds = torch.tensor([0, 1, 1, 2, 2, 2])
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    cm = confusion_matrix(preds, targets, 3)
    assert cm.dtype == torch.int64
    assert cm.tolist() == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]
    assert confusion_matrix(torch.tensor([0]), torch.tensor([0]), 4).shape == (4, 4)


def test_precision_recall_f1_with_zero_division():
    cm = torch.tensor([[2, 0, 0], [1, 1, 0], [0, 0, 0]])
    p, r, f1 = precision_recall_f1(cm)
    assert p.tolist() == pytest.approx([2 / 3, 1.0, 0.0])
    assert r.tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert f1[0].item() == pytest.approx(0.8)
    assert torch.isfinite(f1).all()


def test_match_detections():
    ref = [(1.0, "marvin"), (3.0, "one"), (5.0, "stop")]
    pred = [(1.2, "marvin"), (1.3, "marvin"), (3.1, "two"), (5.4, "stop"), (9.0, "one")]
    r = match_detections(pred, ref, tolerance_s=0.5)
    assert (r.hits, r.misses, r.false_alarms) == (2, 1, 3)


def test_match_detections_tolerance_is_inclusive():
    r = match_detections([(1.5, "a")], [(1.0, "a")], tolerance_s=0.5)
    assert (r.hits, r.misses, r.false_alarms) == (1, 0, 0)


def test_false_alarms_per_hour():
    assert false_alarms_per_hour(3, 1800.0) == pytest.approx(6.0)
