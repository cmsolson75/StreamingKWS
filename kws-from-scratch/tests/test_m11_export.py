"""M11 验收测试：ONNX 导出。不要修改。需要先 `uv sync --extra export`。"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("onnxruntime")

from kws.export import export_onnx, max_abs_diff_vs_onnx
from kws.model import TCResNet8


def test_export_and_compare(tmp_path):
    torch.manual_seed(0)
    model = TCResNet8(40, 12, dropout=0.5).eval()
    path = export_onnx(model, torch.randn(1, 40, 101), tmp_path / "m.onnx")
    assert path.exists()
    assert max_abs_diff_vs_onnx(model, path, torch.randn(1, 40, 101)) < 1e-4
    # batch 维可变
    assert max_abs_diff_vs_onnx(model, path, torch.randn(3, 40, 101)) < 1e-4
