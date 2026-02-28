import torch
import torchaudio
from pathlib import Path

from .configs import Config
from typing import Protocol


class Augmentation(Protocol):
    def apply(self, x: torch.Tensor) -> torch.Tensor: ...


class RandomNoiseMixIn(Augmentation):
    """
    Mixes random noise clips into audio at random SNR
    Expects mono audio tensors of shape (1, num_samples)
    """

    def __init__(self, cfg: Config, eps: float = 1e-8):
        self.cfg = cfg
        self.path = Path(cfg.noise_path)
        self.data: list[torch.Tensor] = self.load_data()
        if len(self.data) == 0:
            raise ValueError("Random noise folder empty")

        self.eps = eps
        self.generator = torch.Generator().manual_seed(self.cfg.seed)

    def load_data(self) -> list[torch.Tensor]:
        data = []
        for file in self.path.rglob("*.wav"):
            if not file.exists() or file.is_dir():
                continue
            audio, sr = torchaudio.load(file)
            # Ensure all audio is the global sample rate
            if sr != self.cfg.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.cfg.sample_rate)
            # ensure shape and mono
            if audio.ndim != 2:
                audio = audio.unsqueeze(0)
            if audio.shape[0] != 1:  # sum to mono
                audio = audio.mean(dim=0, keepdim=True)
            data.append(audio)
        return data

    def _get_random_noise_file(self) -> torch.Tensor:
        idx = torch.randint(0, len(self.data), (1,), generator=self.generator).item()
        return self.data[idx]

    def _random_shape_match(self, noise: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        duration = x.shape[1]
        if noise.shape[1] < duration:
            tiles = [noise]
            remaining = duration - noise.shape[1]
            while remaining > 0:
                chunk = noise[:, :remaining]
                tiles.append(chunk)
                remaining -= chunk.shape[1]
            return torch.cat(tiles, dim=1)

        start_loc = torch.randint(
            0, noise.shape[1] - duration, (1,), generator=self.generator
        )
        return noise[:, start_loc : start_loc + duration]

    def _get_gain_factor(self, noise: torch.Tensor, x: torch.Tensor) -> float:
        snr = torch.empty(1).uniform_(
            self.cfg.min_snr_db, self.cfg.max_snr_db, generator=self.generator
        )
        noise_power = torch.mean(noise**2) + self.eps
        inp_power = torch.mean(x**2)

        target_power = inp_power / (10 ** (snr / 10.0))
        return torch.sqrt(target_power / noise_power).item()

    def _loudness_normalization(self, x: torch.Tensor) -> torch.Tensor:
        peak_amp = torch.max(torch.abs(x)).item()
        if peak_amp > 1.0:
            x = x / peak_amp
        return torch.clip(x, -1.0, 1.0)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.rand(1, generator=self.generator).item()
        if p < self.cfg.prob_noise_mix_in:
            noise = self._get_random_noise_file()
            noise = self._random_shape_match(noise, x)

            gain_factor = self._get_gain_factor(noise, x)
            output = x + noise * gain_factor
            return self._loudness_normalization(output)
        else:
            return x


class RandomGain(Augmentation): ...


# More of audio augmentations
class AugmentationPipeline:
    def __init__(self, augmentations: list[Augmentation]):
        self.augmentations = augmentations

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        for augmentation in self.augmentations:
            x = augmentation.apply(x)
        return x


def build_augmentation_pipeline(
    augmentations: list[Augmentation],
) -> AugmentationPipeline:
    return AugmentationPipeline(augmentations)
