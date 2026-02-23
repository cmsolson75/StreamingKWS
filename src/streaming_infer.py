import numpy as np
import threading
import time
import sounddevice as sd
import torch
from torch import nn
import torch.nn.functional as F
from .configs import Config, InferConfig
from .model import AudioClassifier
from .transforms import DbMelSpec, AudioTransform
import json
from pathlib import Path
from safetensors.torch import load_file

from typing import Dict, Tuple


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
    def __init__(self, model: AudioClassifier, cfg: Config):
        super().__init__()
        self.model = model
        model.eval()

        self.cfg = cfg

        self.transform = AudioTransform(cfg)
        self.db_mel_spec = DbMelSpec(cfg)

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
    model = AudioClassifier(len(cfg.subset)).to(cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class AudioStream:
    def __init__(
        self,
        samplerate,
        channels,
        buffer_size,
        inference_runner: InferenceRunner,
        labels: list,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.buffer_size = buffer_size
        self.buffer = CircularBuffer(self.buffer_size)
        self.inference_runner = inference_runner
        self.labels = labels
        self.silence_threshold = 0.01
        self.threshold = 0.7
        self.cooldown = time.time()
        self.cooldown_time = 1.0

    def can_trigger(self):
        return time.time() >= self.cooldown

    def update_cooldown(self):
        self.cooldown = time.time() + self.cooldown_time

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.buffer.put_many(indata[:, 0])

    def process_audio(self):
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self.callback,
            dtype="float32",
            blocksize=0,
        ):
            try:
                while True:
                    time.sleep(0.2)
                    data = self.buffer.get()
                    if len(data) >= self.buffer_size:
                        probs = self.inference_runner(data)
                        probs = probs.squeeze(0)
                        highest_prob_idx = torch.argmax(probs)
                        if probs[highest_prob_idx] > self.threshold:
                            if self.can_trigger():
                                label = self.labels[highest_prob_idx.item()]
                                print(f"Prediction: {label}")
                                self.update_cooldown()
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
        16_000, 1, buffer_size=16_000, inference_runner=inference_runner, labels=labels
    )
    audio_stream.process_audio()
