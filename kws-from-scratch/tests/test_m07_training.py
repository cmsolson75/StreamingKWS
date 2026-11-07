"""M7 验收测试：优化器、学习率计划、EMA、训练步。不要修改。"""

from __future__ import annotations

import copy
import itertools
import math

import pytest
import torch
from torch import nn

from kws.ema import EMA
from kws.features import LogMelSpectrogram
from kws.model import TCResNet8
from kws.optim import SGD, AdamW, clip_grad_norm
from kws.schedule import build_scheduler, warmup_cosine
from kws.train import evaluate, parse_args, train_step


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 3))


def _run(model, opt, steps=50):
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(16, 5, generator=g)
        y = torch.randint(0, 3, (16,), generator=g)
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


# ------------------------------------------------------------- optimizers
def test_sgd_matches_torch():
    a, b = _tiny_model(), _tiny_model()
    ours = _run(a, SGD(a.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3))
    ref = _run(b, torch.optim.SGD(b.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3))
    for p, q in zip(ours, ref):
        assert torch.allclose(p, q, atol=1e-6)


def test_adamw_matches_torch():
    a, b = _tiny_model(), _tiny_model()
    ours = _run(a, AdamW(a.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05))
    ref = _run(
        b,
        torch.optim.AdamW(b.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05),
    )
    for p, q in zip(ours, ref):
        assert torch.allclose(p, q, atol=1e-6)


def test_adamw_state_dict_round_trip():
    m = _tiny_model()
    opt = AdamW(m.parameters(), lr=1e-2)
    _run(m, opt, steps=3)
    sd = copy.deepcopy(opt.state_dict())
    opt2 = AdamW(m.parameters(), lr=1e-2)
    opt2.load_state_dict(sd)
    assert len(opt2.state) == len(list(m.parameters()))


def test_clip_grad_norm_matches_torch():
    a, b = _tiny_model(), _tiny_model()
    x = torch.randn(4, 5)
    a(x).pow(2).sum().mul(100).backward()
    b(x).pow(2).sum().mul(100).backward()
    n1 = clip_grad_norm(a.parameters(), 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
    assert torch.allclose(n1, n2)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)


# ------------------------------------------------------------- schedule
def test_warmup_cosine_shape():
    W, M = 10, 110
    assert warmup_cosine(0, W, M) == pytest.approx(1 / W)
    assert warmup_cosine(W - 1, W, M) == pytest.approx(1.0)
    assert warmup_cosine(W, W, M) == pytest.approx(1.0)
    assert warmup_cosine(W + 50, W, M) == pytest.approx(0.5)
    assert warmup_cosine(M, W, M) == pytest.approx(0.0, abs=1e-9)
    assert warmup_cosine(M + 100, W, M, min_ratio=0.1) == pytest.approx(0.1)
    assert warmup_cosine(W + 50, W, M, min_ratio=0.2) == pytest.approx(0.6)
    assert warmup_cosine(0, 0, 100) == pytest.approx(1.0)
    values = [warmup_cosine(s, W, M) for s in range(W, M)]
    assert all(a >= b for a, b in itertools.pairwise(values))


def test_build_scheduler_drives_lr():
    m = _tiny_model()
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    sched = build_scheduler(opt, warmup_steps=4, max_steps=20)
    for step in range(10):
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5 * warmup_cosine(step, 4, 20))
        opt.step()
        sched.step()


# ------------------------------------------------------------- EMA
def test_ema_update_formula_and_buffers():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    ema = EMA(model, decay=0.9)
    assert ema.model is not model
    assert not any(p.requires_grad for p in ema.model.parameters())
    assert not ema.model.training
    old = [p.detach().clone() for p in ema.model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    model.train()
    model(torch.randn(8, 4))  # 更新 BN 的 running stats
    ema.update(model)
    for o, e, cur in zip(old, ema.model.parameters(), model.parameters()):
        assert torch.allclose(e, 0.9 * o + 0.1 * cur, atol=1e-6)
    assert torch.allclose(ema.model[1].running_mean, model[1].running_mean)


def test_ema_state_dict_and_copy_to():
    model = _tiny_model()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3)
    ema.update(model)
    restored = EMA(_tiny_model(seed=5), decay=0.1)
    restored.load_state_dict(copy.deepcopy(ema.state_dict()))
    target = _tiny_model(seed=9)
    restored.copy_to(target)
    for p, q in zip(target.parameters(), ema.model.parameters()):
        assert torch.equal(p, q)


# ------------------------------------------------------------- train loop
def test_parse_args():
    assert parse_args(["--config", "c.yaml", "train.lr=0.1", "env.seed=3"]) == (
        "c.yaml",
        ["train.lr=0.1", "env.seed=3"],
    )
    with pytest.raises(SystemExit):
        parse_args(["train.lr=0.1"])


def test_train_step_overfits_one_batch():
    torch.manual_seed(0)
    feats = LogMelSpectrogram(16000, 400, 160, 40)
    model = TCResNet8(40, 4, dropout=0.0)
    x = torch.randn(8, 16000) * 0.1
    y = torch.arange(8) % 4
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    first = train_step(model, feats, (x, y), opt, clip_norm=10.0)
    assert set(first) >= {"loss", "acc", "grad_norm"}
    assert all(isinstance(v, float) for v in first.values())
    for _ in range(80):
        last = train_step(model, feats, (x, y), opt)
    assert last["loss"] < 0.1 * first["loss"]
    assert last["acc"] == 1.0
    assert math.isfinite(last["grad_norm"])


def test_evaluate_weights_by_samples_and_restores_mode():
    class Const(nn.Module):
        def forward(self, x):
            return torch.tensor([[10.0, 0.0]]).expand(x.shape[0], 2)

    model = Const().train()
    loader = [
        (torch.zeros(3, 10), torch.tensor([0, 0, 0])),
        (torch.zeros(1, 10), torch.tensor([1])),
    ]
    out = evaluate(model, nn.Identity(), loader)
    assert out["acc"] == pytest.approx(0.75)  # 不是 (1.0 + 0.0) / 2
    assert model.training
