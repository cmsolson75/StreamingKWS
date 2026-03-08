from enum import Enum, auto
import time

import numpy as np
import sounddevice as sd
import torch

from ..configs import InferConfig
from .buffer import CircularBuffer
from .runner import InferenceRunner


class AppState(Enum):
    IDLE = auto()
    LISTENING = auto()


class AudioStream:
    def __init__(
        self,
        config: InferConfig,
        inference_runner: InferenceRunner,
        labels: list,
        output: str | None = None,
    ):
        self.cfg = config
        self.buffer = CircularBuffer(self.cfg.buffer_size)
        self.inference_runner = inference_runner
        self.labels = labels

        self.cooldown = time.time()
        self.state = AppState.IDLE
        self.numbers = []
        self.output_path = output

    def write_file(self) -> None:
        if self.output_path is None or not self.numbers:
            return
        with open(self.output_path, "a") as f:
            f.write("".join(str(num) for num in self.numbers) + "\n")

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
        if label == self.cfg.stopword:
            self.state = AppState.IDLE
            print("Stopping...")
            if self.numbers:
                str_out = "".join(str(num) for num in self.numbers)
                print(f"Recorded Numbers: {str_out}")
            self.write_file()
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
