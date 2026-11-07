"""M2 验收测试：音频 I/O。不要修改。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf
import torch

from kws.audio import db_to_amplitude, load_audio, pad_or_crop, rms_db


def test_load_audio_mono_float32(tmp_path):
    t = np.arange(16000) / 16000
    sf.write(tmp_path / "a.wav", 0.5 * np.sin(2 * np.pi * 440 * t), 16000, subtype="PCM_16")
    wav = load_audio(tmp_path / "a.wav", 16000)
    assert wav.shape == (16000,)
    assert wav.dtype == torch.float32
    assert wav.abs().max() == pytest.approx(0.5, abs=1e-3)


def test_load_audio_stereo_is_averaged(tmp_path):
    left = np.full(800, 0.2, dtype=np.float32)
    right = np.full(800, 0.6, dtype=np.float32)
    sf.write(tmp_path / "s.wav", np.stack([left, right], axis=1), 16000, subtype="FLOAT")
    wav = load_audio(tmp_path / "s.wav", 16000)
    assert wav.shape == (800,)
    assert torch.allclose(wav, torch.full((800,), 0.4), atol=1e-6)


def test_load_audio_resamples(tmp_path):
    sf.write(tmp_path / "r.wav", np.zeros(8000, dtype=np.float32), 8000)
    assert load_audio(tmp_path / "r.wav", 16000).shape == (16000,)


def test_pad_left():
    x = torch.arange(1, 4, dtype=torch.float32)
    assert pad_or_crop(x, 5, "left").tolist() == [1, 2, 3, 0, 0]


def test_pad_center_and_crop_center():
    x = torch.arange(1, 4, dtype=torch.float32)
    assert pad_or_crop(x, 6, "center").tolist() == [0, 1, 2, 3, 0, 0]
    y = torch.arange(10, dtype=torch.float32)
    assert pad_or_crop(y, 4, "center").tolist() == [3, 4, 5, 6]


def test_crop_left_and_exact_length():
    y = torch.arange(10, dtype=torch.float32)
    assert pad_or_crop(y, 4, "left").tolist() == [0, 1, 2, 3]
    assert torch.equal(pad_or_crop(y, 10, "random"), y)


def test_pad_or_crop_keeps_leading_dims():
    x = torch.randn(2, 3, 50)
    assert pad_or_crop(x, 80, "random").shape == (2, 3, 80)
    assert pad_or_crop(x, 20, "random").shape == (2, 3, 20)


def test_random_mode_is_reproducible_and_varies():
    y = torch.arange(1000, dtype=torch.float32)
    a = pad_or_crop(y, 100, "random", generator=torch.Generator().manual_seed(0))
    b = pad_or_crop(y, 100, "random", generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)
    # 裁出来的必须是连续的一段
    assert torch.equal(a, torch.arange(a[0].item(), a[0].item() + 100))
    g = torch.Generator().manual_seed(1)
    starts = {pad_or_crop(y, 100, "random", generator=g)[0].item() for _ in range(20)}
    assert len(starts) > 1


def test_random_pad_keeps_signal_intact():
    x = torch.arange(1, 6, dtype=torch.float32)
    out = pad_or_crop(x, 12, "random", generator=torch.Generator().manual_seed(3))
    nz = out.nonzero().flatten()
    assert torch.equal(out[nz[0] : nz[0] + 5], x)
    assert out.sum() == x.sum()


def test_rms_db_of_unit_sine():
    t = torch.arange(16000) / 16000
    sine = torch.sin(2 * math.pi * 100 * t)
    assert rms_db(sine).item() == pytest.approx(20 * math.log10(1 / math.sqrt(2)), abs=0.01)
    assert rms_db(torch.ones(2, 100)).shape == (2,)
    assert torch.isfinite(rms_db(torch.zeros(100)))


def test_db_to_amplitude():
    assert db_to_amplitude(0.0) == pytest.approx(1.0)
    assert db_to_amplitude(20.0) == pytest.approx(10.0)
    assert db_to_amplitude(-6.0) == pytest.approx(0.501, abs=1e-3)
