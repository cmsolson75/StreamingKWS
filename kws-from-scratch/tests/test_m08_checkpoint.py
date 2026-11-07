"""M8 验收测试：存档与严格恢复。不要修改。

核心测试 test_resume_is_exact：
    A：连续训练 40 步
    B：训练 20 步 → 存档 → 把所有对象和随机状态都换成“新进程”的样子 → 读档 → 再训练 20 步
    要求 A 和 B 的最终模型、EMA、学习率、采样器位置完全一致。
训练过程故意用到了：dropout（torch 随机）、numpy 随机、Python random、采样器 generator。
漏存任何一个，这个测试都会失败。
"""

from __future__ import annotations

import json
import random

import numpy as np
import torch
from torch import nn

from kws.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    read_pointer,
    restore_rng_state,
    save_checkpoint,
    write_pointer,
)
from kws.ema import EMA
from kws.optim import AdamW
from kws.sampler import GroupWeightedSampler
from kws.schedule import build_scheduler

TOTAL, HALF = 40, 20
DATA = torch.randn(64, 6, generator=torch.Generator().manual_seed(0))
LABELS = torch.randint(0, 3, (64,), generator=torch.Generator().manual_seed(1))


def _objects(seed: int):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 3))
    opt = AdamW(model.parameters(), lr=1e-2)
    sched = build_scheduler(opt, warmup_steps=5, max_steps=TOTAL)
    ema = EMA(model, decay=0.9)
    sampler = GroupWeightedSampler(
        [32, 32], [0.5, 0.5], num_samples=10_000, generator=torch.Generator().manual_seed(seed)
    )
    return model, opt, sched, ema, sampler


def _train(model, opt, sched, ema, sampler_iter, steps):
    model.train()
    for _ in range(steps):
        idx = torch.tensor([next(sampler_iter) for _ in range(8)])
        x = DATA[idx] + 0.01 * torch.from_numpy(np.random.randn(8, 6)).float()
        x = x * random.uniform(0.9, 1.1)
        loss = nn.functional.cross_entropy(model(x), LABELS[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        ema.update(model)


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_rng_state_round_trip(tmp_path):
    _seed_all(0)
    state = capture_rng_state()
    a = (random.random(), np.random.rand(), torch.rand(()).item())
    _seed_all(123)
    restore_rng_state(state)
    assert (random.random(), np.random.rand(), torch.rand(()).item()) == a
    torch.save(state, tmp_path / "rng.pt")
    torch.load(tmp_path / "rng.pt", weights_only=True)


def test_resume_is_exact(tmp_path):
    # A：一口气训练
    _seed_all(0)
    model_a, opt_a, sched_a, ema_a, sampler_a = _objects(0)
    _train(model_a, opt_a, sched_a, ema_a, iter(sampler_a), TOTAL)

    # B 前半段
    _seed_all(0)
    model_b, opt_b, sched_b, ema_b, sampler_b = _objects(0)
    it_b = iter(sampler_b)
    _train(model_b, opt_b, sched_b, ema_b, it_b, HALF)
    ckpt = tmp_path / "step_20.pt"
    save_checkpoint(
        ckpt,
        step=HALF,
        model=model_b,
        optimizer=opt_b,
        scheduler=sched_b,
        ema=ema_b,
        sampler=sampler_b,
    )
    torch.load(ckpt, weights_only=True)  # 必须能安全读取

    # 模拟新进程：全新的对象、不同的初始化、被打乱的全局随机状态
    _seed_all(777)
    model_c, opt_c, sched_c, ema_c, sampler_c = _objects(777)
    info = load_checkpoint(
        ckpt, model=model_c, optimizer=opt_c, scheduler=sched_c, ema=ema_c, sampler=sampler_c
    )
    assert info["step"] == HALF
    _train(model_c, opt_c, sched_c, ema_c, iter(sampler_c), TOTAL - HALF)

    assert opt_c.param_groups[0]["lr"] == opt_a.param_groups[0]["lr"]
    for p, q in zip(model_a.parameters(), model_c.parameters()):
        assert torch.allclose(p, q, atol=1e-7)
    for p, q in zip(ema_a.model.parameters(), ema_c.model.parameters()):
        assert torch.allclose(p, q, atol=1e-7)
    assert next(iter(sampler_a)) == next(iter(sampler_c))


def test_load_for_eval_only(tmp_path):
    model, opt, _sched, _ema, _sampler = _objects(0)
    save_checkpoint(
        tmp_path / "c.pt", step=3, model=model, optimizer=opt, extra={"labels": ["a", "b"]}
    )
    fresh = _objects(5)[0]
    info = load_checkpoint(tmp_path / "c.pt", model=fresh, restore_rng=False)
    assert info["extra"] == {"labels": ["a", "b"]}
    for p, q in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p, q)


def test_atomic_save_leaves_no_temp_files(tmp_path):
    model, opt, *_ = _objects(0)
    save_checkpoint(tmp_path / "c.pt", step=1, model=model, optimizer=opt)
    save_checkpoint(tmp_path / "c.pt", step=2, model=model, optimizer=opt)  # 覆盖
    assert [p.name for p in tmp_path.iterdir()] == ["c.pt"]


def test_pointers(tmp_path):
    assert read_pointer(tmp_path, "latest") is None
    (tmp_path / "step_10.pt").write_bytes(b"x")
    write_pointer(tmp_path, "latest", tmp_path / "step_10.pt", step=10)
    assert read_pointer(tmp_path, "latest") == (tmp_path / "step_10.pt").resolve()
    data = json.loads((tmp_path / "latest.json").read_text())
    assert data["step"] == 10
    assert not any(str(tmp_path) in str(v) for v in data.values()), (
        "指针里存相对路径，目录搬家后才不会失效"
    )
