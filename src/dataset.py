import torchaudio
import torch
from torch.utils.data import Dataset
from .configs import Config
from pathlib import Path
from typing import Literal
import json
from .transforms import AudioTransform
from .augmentations import AugmentationPipeline


def load_labels(
    cfg: Config,
    write_json: bool = False,
    labels_path: str = "labels.json",
):
    path = Path(cfg.data.path)
    subset = set(cfg.data.subset) if cfg.data.subset is not None else set()
    labels = sorted(
        [
            p.name
            for p in path.iterdir()
            if p.is_dir() and p.name[0] != "_" and p.name in subset
        ]
    )
    labels.append("<UNKNOWN>")
    labels.append("<SILENCE>")
    num_classes = len(labels)
    label_to_idx = {w: i for i, w in enumerate(labels)}

    if write_json:
        labels_path = Path(labels_path)
        with labels_path.open("w") as f:
            f.write(json.dumps(labels))

    return labels, label_to_idx, num_classes


class SpeechCommands(Dataset):
    def __init__(
        self,
        cfg: Config,
        split: str,
        augmentation_pipeline: AugmentationPipeline | None = None,
    ):
        self.cfg: Config = cfg
        self.subset = (
            set(self.cfg.data.subset) if self.cfg.data.subset is not None else set()
        )
        self.path: Path = Path(self.cfg.data.path)
        self.split: str = split
        self.val_split = self.open_split("val")
        self.test_split = self.open_split("test")
        self.dataset = self._load_dataset()
        self.transform = AudioTransform(cfg)

        self.augmentation_pipeline = augmentation_pipeline

        self.cache = {}  # this will be useful for reducing compute: Cache before augs

    def _load_dataset(self):
        self.labels, self.stoi, self.num_classes = load_labels(
            cfg=self.cfg, write_json=True, labels_path="labels.json"
        )
        dataset = []
        for label in self.labels:
            label_path = self.path / label
            for file in label_path.rglob("*.wav"):
                file_id = f"{label}/{file.name}"
                if self.in_split(file_id):
                    dataset.append((str(file), label))
        return dataset

    def open_split(self, split: Literal["val", "test"]) -> set:
        split_files = {"val": "validation_list.txt", "test": "testing_list.txt"}
        split_file = split_files.get(split)
        split_path = self.path / split_file
        with split_path.open("r") as file:
            data = set(sorted(list(file.read().splitlines())))
        return data

    def in_split(self, file_id: str):
        if (
            self.split == "train"
            and file_id not in self.val_split
            and file_id not in self.test_split
        ):
            return True
        if self.split == "test" and file_id in self.test_split:
            return True
        if self.split == "val" and file_id in self.val_split:
            return True
        return False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        audio_path, label = self.dataset[idx]
        if audio_path not in self.cache:
            wav, sr = torchaudio.load(audio_path)
            audio = self.transform(wav, sr)
            self.cache[audio_path] = audio
        else:
            audio = self.cache.get(audio_path)

        if (
            self.cfg.augment.use_augmentations
            and self.augmentation_pipeline is not None
        ):
            audio = self.augmentation_pipeline.apply(audio)

        return audio, self.stoi[label]


class OOVDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        split: str,
        augmentation_pipeline: AugmentationPipeline | None = None,
    ):
        self.cfg: Config = cfg
        self.split = split
        self.subset = (
            set(self.cfg.data.subset) if self.cfg.data.subset is not None else set()
        )
        self.path: Path = Path(self.cfg.data.path)
        _, self.stoi, _ = load_labels(self.cfg)
        self.val_split = self.open_split("val")
        self.test_split = self.open_split("test")
        self.dataset = self._load_dataset()
        self.transform = AudioTransform(self.cfg)
        self.augmentation_pipeline = augmentation_pipeline
        self.cache = {}

    def open_split(self, split: Literal["val", "test"]) -> set:
        split_files = {"val": "validation_list.txt", "test": "testing_list.txt"}
        split_file = split_files.get(split)
        split_path = self.path / split_file
        with split_path.open("r") as file:
            data = set(sorted(list(file.read().splitlines())))
        return data

    def in_split(self, file_id: str):
        if (
            self.split == "train"
            and file_id not in self.val_split
            and file_id not in self.test_split
        ):
            return True
        if self.split == "test" and file_id in self.test_split:
            return True
        if self.split == "val" and file_id in self.val_split:
            return True
        return False

    def _load_dataset(self):
        subset = set(self.subset) if self.subset is not None else set()
        labels = sorted(
            [
                p.name
                for p in self.path.iterdir()
                if p.is_dir() and p.name[0] != "_" and p.name not in subset
            ]
        )
        dataset = []
        for label in labels:
            label_path = self.path / label
            for file in label_path.rglob("*.wav"):
                file_id = f"{label}/{file.name}"
                if self.in_split(file_id):
                    dataset.append((str(file), "<UNKNOWN>"))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_path, label = self.dataset[idx]
        if audio_path not in self.cache:
            wav, sr = torchaudio.load(audio_path)
            audio = self.transform(wav, sr)
            self.cache[audio_path] = audio
        else:
            audio = self.cache.get(audio_path)
        if (
            self.cfg.augment.use_augmentations
            and self.augmentation_pipeline is not None
        ):
            audio = self.augmentation_pipeline.apply(audio)

        return audio, self.stoi[label]


class SyntheticSilenceDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        size: int,
        augmentation_pipeline: AugmentationPipeline | None = None,
    ):
        self.cfg = cfg
        self.num_samples = int(
            self.cfg.preprocess.clip_length * self.cfg.preprocess.sample_rate
        )
        _, self.stoi, _ = load_labels(self.cfg)
        self.size = size
        self.augmentation_pipeline = augmentation_pipeline

    def generate_silence(self, noise_floor=-60):
        amplitude = 10 ** (noise_floor / 20)
        return torch.randn((1, self.num_samples)) * amplitude

    def __len__(self):
        return self.size

    def __getitem__(self, _) -> torch.Tensor:
        silence = self.generate_silence()
        if (
            self.cfg.augment.use_augmentations
            and self.augmentation_pipeline is not None
        ):
            silence = self.augmentation_pipeline.apply(silence)
        return silence, self.stoi["<SILENCE>"]


class BackgroundNoiseDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        size: int,
        augmentation_pipeline: AugmentationPipeline | None = None,
    ):
        self.cfg = cfg
        self.size = size
        self.path = Path(cfg.augment.noise_path)
        self.data: list[torch.Tensor] = self._load_noise()
        _, self.stoi, _ = load_labels(self.cfg)
        if len(self.data) == 0:
            raise ValueError("Random noise folder empty")

        self.augmentation_pipeline = augmentation_pipeline
        self.generator = torch.Generator().manual_seed(self.cfg.env.seed)

        self.expected_len = int(
            self.cfg.preprocess.sample_rate * self.cfg.preprocess.clip_length
        )

    def _load_noise(self) -> list[torch.Tensor]:
        data = []
        for file in self.path.rglob("*.wav"):
            if not file.exists() or file.is_dir():
                continue
            audio, sr = torchaudio.load(file)
            # Ensure all audio is the global sample rate
            if sr != self.cfg.preprocess.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sr, self.cfg.preprocess.sample_rate
                )
            # ensure shape and mono
            if audio.ndim != 2:
                audio = audio.unsqueeze(0)
            if audio.shape[0] != 1:  # sum to mono
                audio = audio.mean(dim=0, keepdim=True)
            data.append(audio)
        return data

    def _random_crop(self, noise: torch.Tensor) -> torch.Tensor:
        start_loc = torch.randint(
            0, noise.shape[1] - self.expected_len, (1,), generator=self.generator
        )
        return noise[:, start_loc : start_loc + self.expected_len]

    def _get_random_noise(self) -> torch.Tensor:
        idx = torch.randint(0, len(self.data), (1,), generator=self.generator).item()
        return self.data[idx]

    def __len__(self):
        return self.size

    def __getitem__(self, _: int) -> torch.Tensor:
        noise = self._get_random_noise()
        noise = self._random_crop(noise)
        if (
            self.cfg.augment.use_augmentations
            and self.augmentation_pipeline is not None
        ):
            noise = self.augmentation_pipeline.apply(noise)
        return noise, self.stoi["<SILENCE>"]


if __name__ == "__main__":
    cfg = Config.from_yaml("configs/config.yaml")
    split = "test"
    sc = SpeechCommands(cfg, split)
    unknown = OOVDataset(cfg, split)
    silence_pool_size = int((len(sc) + len(unknown)) * cfg.sampler.silence_weight)
    silence = SyntheticSilenceDataset(cfg, silence_pool_size)

    breakpoint()
