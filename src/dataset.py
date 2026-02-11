import torch
import torchaudio
from torch.utils.data import Dataset
from .configs import Config
from pathlib import Path
from typing import Literal
import json
from .transforms import AudioTransform


def load_labels(
    cfg: Config,
    write_json: bool = False,
    labels_path: str = "labels.json",
):
    path = Path(cfg.path)
    subset = set(cfg.subset) if cfg.subset is not None else set()
    labels = sorted(
        [
            p.name
            for p in path.iterdir()
            if p.is_dir() and p.name[0] != "_" and p.name in subset
        ]
    )
    num_classes = len(labels)
    label_to_idx = {w: i for i, w in enumerate(labels)}
    if write_json:
        labels_path = Path(labels_path)
        with labels_path.open("w") as f:
            f.write(json.dumps(labels))
    return labels, label_to_idx, num_classes


class SpeechCommands(Dataset):
    def __init__(self, cfg: Config, split: str):
        self.cfg: Config = cfg
        self.subset = set(self.cfg.subset) if self.cfg.subset is not None else set()
        self.path: Path = Path(self.cfg.path)
        self.split: str = split
        self.val_split = self.open_split("val")
        self.test_split = self.open_split("test")
        self.dataset = self._load_dataset()
        self.transform = AudioTransform(cfg)

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

        return audio, self.stoi[label]


if __name__ == "__main__":
    cfg = Config.from_yaml("configs/config.yaml")
    dataset = SpeechCommands(cfg, "train")
