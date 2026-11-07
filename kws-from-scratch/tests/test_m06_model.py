"""M6 验收测试：模型。不要修改。"""

from __future__ import annotations

import pytest
import torch

from kws.config import Config
from kws.model import SimpleCNN, TCResBlock, TCResNet8, build_model, count_parameters


@pytest.mark.parametrize("cls", [SimpleCNN, TCResNet8])
@pytest.mark.parametrize("frames", [101, 51, 17])
def test_output_shape_any_length(cls, frames):
    model = cls(n_mels=64, num_classes=14)
    assert model(torch.randn(3, 64, frames)).shape == (3, 14)


def test_simple_cnn_is_small():
    assert count_parameters(SimpleCNN(64, 14)) < 200_000


def test_tc_block_shapes():
    assert TCResBlock(16, 24, stride=2)(torch.randn(2, 16, 101)).shape == (2, 24, 51)
    assert TCResBlock(24, 24, stride=1)(torch.randn(2, 24, 51)).shape == (2, 24, 51)


def test_tc_resnet8_param_count_matches_paper():
    # 论文 Table：TC-ResNet8 约 66K 参数（40 维特征，12 类，k=1）
    n = count_parameters(TCResNet8(n_mels=40, num_classes=12))
    assert 60_000 < n < 72_000, n


def test_width_mult_scales_params():
    small = count_parameters(TCResNet8(40, 12, width_mult=1.0))
    big = count_parameters(TCResNet8(40, 12, width_mult=1.5))
    assert 1.8 < big / small < 2.6


def test_count_parameters_ignores_frozen():
    m = TCResNet8(40, 12)
    total = count_parameters(m)
    m.requires_grad_(False)
    assert count_parameters(m) == 0 and total > 0


def test_eval_is_deterministic_train_is_not():
    torch.manual_seed(0)
    m = TCResNet8(64, 14, dropout=0.5)
    x = torch.randn(4, 64, 101)
    m.eval()
    assert torch.equal(m(x), m(x))
    m.train()
    assert not torch.equal(m(x), m(x))


def test_gradients_reach_every_parameter():
    m = TCResNet8(64, 14)
    m(torch.randn(4, 64, 101)).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_build_model(config_path):
    cfg = Config.from_yaml(config_path)
    assert isinstance(build_model(cfg.model, 64, 14), TCResNet8)
    assert isinstance(build_model(cfg.with_overrides(["model.name=cnn"]).model, 64, 14), SimpleCNN)
