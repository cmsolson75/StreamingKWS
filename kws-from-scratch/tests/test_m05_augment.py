"""M5 验收测试：数据增强。不要修改。"""

from __future__ import annotations

import pytest
import torch

from kws.augment import Compose, RandomGain, RandomNoise, build_augment, load_noise_bank, mix_at_snr
from kws.config import Config


def _snr_db(clean, mixed):
    noise = mixed - clean
    return (10 * torch.log10(clean.pow(2).mean() / noise.pow(2).mean())).item()


@pytest.mark.parametrize("snr", [-5.0, 0.0, 10.0, 25.0])
def test_mix_at_snr_hits_target(snr):
    g = torch.Generator().manual_seed(0)
    clean = torch.randn(16000, generator=g) * 0.3
    noise = torch.randn(16000, generator=g) * 2.0
    mixed = mix_at_snr(clean, noise, snr)
    assert mixed.shape == clean.shape
    assert _snr_db(clean, mixed) == pytest.approx(snr, abs=0.01)


def test_mix_at_snr_silent_clean_is_finite():
    out = mix_at_snr(torch.zeros(100), torch.randn(100), 10.0)
    assert torch.isfinite(out).all()


def test_random_gain_range_and_probability():
    g = torch.Generator().manual_seed(0)
    aug = RandomGain(-6.0, 6.0, p=1.0, generator=g)
    x = torch.ones(10)
    ratios = torch.stack([aug(x)[0] for _ in range(200)])
    db = 20 * torch.log10(ratios)
    assert db.min() >= -6.0 - 1e-4 and db.max() <= 6.0 + 1e-4
    assert db.std() > 1.0
    never = RandomGain(-6.0, 6.0, p=0.0, generator=g)
    assert torch.equal(never(x), x)


def test_random_gain_reproducible():
    x = torch.randn(100)
    a = RandomGain(-10, 10, 0.5, generator=torch.Generator().manual_seed(3))
    b = RandomGain(-10, 10, 0.5, generator=torch.Generator().manual_seed(3))
    for _ in range(10):
        assert torch.equal(a(x), b(x))


def test_random_noise_snr_in_range():
    g = torch.Generator().manual_seed(0)
    bank = [torch.randn(48000, generator=g), torch.randn(30000, generator=g) * 5]
    aug = RandomNoise(bank, min_snr_db=5.0, max_snr_db=15.0, p=1.0, generator=g)
    clean = torch.randn(16000, generator=g) * 0.2
    for _ in range(20):
        out = aug(clean)
        assert out.shape == clean.shape
        assert 5.0 - 0.05 <= _snr_db(clean, out) <= 15.0 + 0.05
    assert torch.equal(RandomNoise(bank, 5.0, 15.0, p=0.0)(clean), clean)


def test_random_noise_uses_different_segments():
    g = torch.Generator().manual_seed(0)
    bank = [torch.randn(48000, generator=g)]
    aug = RandomNoise(bank, 10.0, 10.0, p=1.0, generator=g)
    clean = torch.ones(1000)
    a, b = aug(clean), aug(clean)
    assert not torch.allclose(a, b)


def test_compose_order():
    calls = []
    c = Compose([lambda x: (calls.append("a"), x + 1)[1], lambda x: (calls.append("b"), x * 2)[1]])
    assert c(torch.tensor(1.0)).item() == 4.0
    assert calls == ["a", "b"]


def test_build_augment(config_path, fake_speech_commands):
    cfg = Config.from_yaml(config_path)
    bank = load_noise_bank(fake_speech_commands / "_background_noise_", 16000)
    assert len(bank) == 2 and all(b.ndim == 1 for b in bank)
    assert build_augment(cfg.with_overrides(["augment.enabled=false"]).augment, bank) is None
    aug = build_augment(cfg.augment, bank)
    assert aug is not None
    assert aug(torch.randn(16000) * 0.1).shape == (16000,)
