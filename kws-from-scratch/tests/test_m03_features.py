"""M3 验收测试：特征提取。不要修改。

这里用 torchaudio 当“标准答案”。你的实现里不能调用 torch.stft 或 torchaudio。
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch
import torchaudio

from kws import features
from kws.features import (
    LogMelSpectrogram,
    SpecAugment,
    frame_signal,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    power_spectrogram,
    power_to_db,
)

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 64


def test_no_forbidden_calls():
    """检查源码：不能用 torch.stft、torchaudio。"""
    tree = ast.parse(inspect.getsource(features))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imports = {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in n.names
    }
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "stft" not in names
    assert not any("torchaudio" in (m or "") for m in modules | imports)


def test_mel_scale_round_trip():
    hz = torch.tensor([0.0, 700.0, 4000.0, 8000.0])
    assert hz_to_mel(torch.tensor(700.0)).item() == pytest.approx(781.17, abs=0.01)
    assert torch.allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-2)


def test_mel_filterbank_matches_torchaudio():
    n_freqs = N_FFT // 2 + 1
    ours = mel_filterbank(n_freqs, N_MELS, SR)
    ref = torchaudio.functional.melscale_fbanks(
        n_freqs, 0.0, SR / 2, N_MELS, SR, norm=None, mel_scale="htk"
    )
    assert ours.shape == (n_freqs, N_MELS)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_frame_signal_shape_and_content():
    x = torch.arange(1000, dtype=torch.float32)
    frames = frame_signal(x, n_fft=100, hop_length=50, center=False)
    assert frames.shape == (19, 100)
    assert torch.equal(frames[1], x[50:150])
    centered = frame_signal(torch.randn(2, SR), N_FFT, HOP, center=True)
    assert centered.shape == (2, 1 + SR // HOP, N_FFT)


def test_power_spectrogram_matches_torch_stft():
    wav = torch.randn(3, SR)
    ours = power_spectrogram(wav, N_FFT, HOP)
    ref = (
        torch.stft(
            wav,
            N_FFT,
            HOP,
            window=torch.hann_window(N_FFT),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        ** 2
    )
    assert ours.shape == ref.shape == (3, N_FFT // 2 + 1, 1 + SR // HOP)
    assert torch.allclose(ours, ref, rtol=1e-4, atol=1e-3)


def test_power_to_db():
    p = torch.tensor([1.0, 10.0, 100.0, 0.0])
    assert torch.allclose(power_to_db(p)[:3], torch.tensor([0.0, 10.0, 20.0]))
    assert power_to_db(p)[3].item() == pytest.approx(-100.0)
    spec = torch.rand(2, 8, 10) * 1e4
    db = power_to_db(spec, top_db=30.0)
    peak = db.amax(dim=(-2, -1), keepdim=True)
    assert (db >= peak - 30.0 - 1e-4).all()


def test_log_mel_matches_torchaudio():
    wav = torch.randn(4, SR) * 0.1
    ours = LogMelSpectrogram(SR, N_FFT, HOP, N_MELS)(wav)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
        norm=None,
        mel_scale="htk",
    )(wav)
    ref = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    assert ours.shape == (4, N_MELS, 101)
    assert torch.allclose(ours, ref, atol=1e-3)


def test_log_mel_uses_buffers():
    m = LogMelSpectrogram(SR, N_FFT, HOP, N_MELS)
    assert len(list(m.parameters())) == 0, "窗函数和滤波器组不应该是可训练参数"
    assert len(list(m.buffers())) >= 2
    assert len(m.state_dict()) >= 2


def test_spec_augment_eval_and_disabled_are_identity():
    spec = torch.randn(2, N_MELS, 101)
    aug = SpecAugment(freq_mask=10, time_mask=20).eval()
    assert torch.equal(aug(spec), spec)
    off = SpecAugment(freq_mask=0, time_mask=0).train()
    assert torch.equal(off(spec), spec)


def test_spec_augment_masks_in_train_mode():
    torch.manual_seed(0)
    spec = torch.randn(8, N_MELS, 101)
    original = spec.clone()
    out = SpecAugment(freq_mask=20, time_mask=30).train()(spec)
    assert out.shape == spec.shape
    assert torch.equal(spec, original), "不能原地修改输入"
    assert not torch.equal(out, spec)
