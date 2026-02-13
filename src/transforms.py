import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from .configs import Config

import torchaudio.functional as AF


class DbMelSpec(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)
        fb = AF.melscale_fbanks(
            n_freqs=cfg.n_fft // 2 + 1,
            f_min=0.0,
            f_max=cfg.sample_rate / 2,
            n_mels=cfg.n_mels,
            sample_rate=cfg.sample_rate,
            norm=None,
            mel_scale="htk",
        )
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.squeeze(1)

        x = x.contiguous()

        X = torch.stft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_len,
            win_length=self.cfg.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )

        power = X.abs() ** 2

        mel = torch.matmul(self.fb.T, power)
        db = AF.amplitude_to_DB(
            mel, multiplier=10, amin=1e-10, db_multiplier=0.0, top_db=80
        )
        mean = db.mean(dim=(-2, -1), keepdim=True)
        std = db.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return ((db - mean) / std).unsqueeze(1)


class AudioTransform(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.expected_length = int(self.cfg.sample_rate * self.cfg.clip_length)

    def forward(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        if wav.ndim != 2:
            wav = wav.unsqueeze(0)

        if wav.shape[0] != 1:  # sum to mono
            wav = wav.mean(dim=0, keepdim=True)

        n = wav.shape[1]
        if n > self.expected_length:
            wav = wav[:, : self.expected_length]
        if n < self.expected_length:
            pad_amt = self.expected_length - n
            wav = F.pad(wav, (0, pad_amt))
        return wav


if __name__ == "__main__":
    audio_example = "/Users/cameronolson/Downloads/T3.wav"
    cfg = Config()
    transform = AudioTransform(cfg)
    wav, sr = torchaudio.load(audio_example)
    # wav, sr = torch.randn(15000), 16_000

    out = transform(wav, sr)
    print(out.shape)
