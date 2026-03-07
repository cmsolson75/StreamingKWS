import numpy as np
import threading
import time
import sounddevice as sd
import torch
from torch import nn
import torch.nn.functional as F
from .configs import Config, InferConfig
from .model import model_factory
from .transforms import DbMelSpec, AudioTransform
import json
from pathlib import Path
from safetensors.torch import load_file

from typing import Dict, Tuple
from enum import Enum, auto


class CircularBuffer:
    def __init__(self, size, dtype=float):
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.head = 0
        self.count = 0
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            if self.count < self.size:
                self.buffer[self.head] = item
                self.count += 1
            else:
                self.buffer[self.head] = item

            self.head = (self.head + 1) % self.size

    def put_many(self, items):
        with self.lock:
            n = len(items)
            first_chunk = min(n, self.size - self.head)
            self.buffer[self.head : self.head + first_chunk] = items[:first_chunk]
            if first_chunk < n:
                self.buffer[: n - first_chunk] = items[first_chunk:]
            self.head = (self.head + n) % self.size
            self.count = min(self.count + n, self.size)

    def get(self):
        with self.lock:
            if self.count == 0:
                return np.array([])

            if self.count < self.size:
                return self.buffer[: self.count]
            else:
                return np.concatenate(
                    (self.buffer[self.head :], self.buffer[: self.head])
                )


class InferenceRunner(nn.Module):
    def __init__(self, model: nn.Module, cfg: Config):
        super().__init__()
        self.model = model
        model.eval()

        self.cfg = cfg

        self.transform = AudioTransform(cfg)
        self.db_mel_spec = DbMelSpec(cfg, augment=False)  # for inference

    @torch.inference_mode()
    def forward(self, x: np.ndarray) -> torch.Tensor:
        """
        return probabilities from model not raw logits
        """
        x = torch.from_numpy(x).to(torch.float32)
        x = self.transform(x, self.cfg.sample_rate)
        x = self.db_mel_spec(x)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        return probs


def load_labels(label_path: str):
    path = Path(label_path)
    with path.open("r") as f:
        return json.load(f)


def load_cfg_model_state(
    model_path: str, device: str | None = None
) -> Tuple[Config, Dict[str, torch.Tensor]]:
    path = Path(model_path)
    if path.suffix == ".json":
        json_data = json.loads(path.read_text())["path"]
        path = path.parent / json_data
    base_path = path.parent.parent

    state_dict = load_file(path / "model.safetensors")
    json_config = base_path / "config.resolved.json"
    device_overwrite = None
    if device is not None:
        device_overwrite = [f"device={device}"]
    return Config.from_json(str(json_config)).with_overrides(
        device_overwrite
    ), state_dict


def load_model(cfg: Config, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    num_classes = len(cfg.data.subset) + 2
    model = model_factory(cfg.model.name, num_classes, cfg.train.dropout).to(
        cfg.train.device
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


class AppState(Enum):
    IDLE = auto()
    LISTENING = auto()


class AudioStream:
    def __init__(
        self,
        config: InferConfig,
        inference_runner: InferenceRunner,
        labels: list,
    ):
        self.cfg = config
        self.buffer = CircularBuffer(self.cfg.buffer_size)
        self.inference_runner = inference_runner
        self.labels = labels

        self.cooldown = time.time()
        # self.cooldown_time = 1.0
        self.state = AppState.IDLE

        self.numbers = []

    def can_trigger(self):
        return time.time() >= self.cooldown

    def update_cooldown(self):
        self.cooldown = time.time() + self.cfg.cooldown_time

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.buffer.put_many(indata[:, 0])

    def detect_keyword(self, data: np.ndarray) -> str | None:
        probs = self.inference_runner(data).squeeze(0)
        max_idx = torch.argmax(probs)
        if probs[max_idx] > self.cfg.threshold and self.can_trigger():
            self.update_cooldown()
            return self.labels[max_idx.item()]
        return None

    def _handle_idle(self, label: str):
        if label == self.cfg.wakeword:
            self.state = AppState.LISTENING
            print("Listening")

    def _handle_listening(self, label: str):
        if label == "stop":
            self.state = AppState.IDLE
            print("Stopping...")
            print(",".join(str(num) for num in self.numbers))
            self.numbers = []
            return

        num = self.cfg.number_map.get(label)
        if num is not None:
            self.numbers.append(num)
            print(f"Number: {num}")

    def process_audio(self):
        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            callback=self.callback,
            dtype="float32",
            blocksize=0,
        ):
            try:
                while True:
                    time.sleep(0.2)
                    data = self.buffer.get()
                    if len(data) < self.cfg.buffer_size:
                        continue
                    label = self.detect_keyword(data)
                    if label is None:
                        continue

                    if self.state == AppState.IDLE:
                        self._handle_idle(label)
                    elif self.state == AppState.LISTENING:
                        self._handle_listening(label)
            except KeyboardInterrupt:
                print("Stopped")


if __name__ == "__main__":
    infer_cfg = InferConfig.from_yaml("configs/infer.yaml")
    device = infer_cfg.device
    cfg, state_dict = load_cfg_model_state(infer_cfg.model, device)
    model = load_model(cfg, state_dict)

    inference_runner = InferenceRunner(model, cfg).to(device)

    labels = load_labels("labels.json")
    audio_stream = AudioStream(
        config=infer_cfg, inference_runner=inference_runner, labels=labels
    )
    audio_stream.process_audio()
