"""测试共用的 fixture。

不要改这个文件，但值得读懂它：
- fixture 是什么、scope 是什么意思
- 为什么测试要用一份“假的、很小的” Speech Commands，而不是真数据集
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000

FAKE_KEYWORDS = ["marvin", "stop", "one", "two"]
FAKE_UNKNOWN_WORDS = ["bed", "cat"]
FAKE_SPEAKERS = [f"spk{i}" for i in range(6)]
VAL_SPEAKERS = {"spk4"}
TEST_SPEAKERS = {"spk5"}


@pytest.fixture
def config_path() -> Path:
    return REPO_ROOT / "configs" / "config.yaml"


@pytest.fixture(scope="session")
def fake_speech_commands(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """一个结构和真实数据集相同的迷你版本：

    - 6 个词（4 个关键词 + 2 个其他词），每个词 6 个说话人各 1 条
    - spk4 的录音进 validation_list.txt，spk5 的进 testing_list.txt
    - 每条录音 1 秒，但每个词的 spk0 只有 0.75 秒（用来测补齐）
    - _background_noise_ 下 2 个 3 秒的噪声文件
    """
    root = tmp_path_factory.mktemp("speech_commands")
    rng = np.random.default_rng(0)
    val_lines, test_lines = [], []
    for word_idx, word in enumerate(FAKE_KEYWORDS + FAKE_UNKNOWN_WORDS):
        (root / word).mkdir()
        for spk in FAKE_SPEAKERS:
            n = int(0.75 * SAMPLE_RATE) if spk == "spk0" else SAMPLE_RATE
            t = np.arange(n) / SAMPLE_RATE
            wav = 0.3 * np.sin(2 * np.pi * (200 + 100 * word_idx) * t)
            wav += 0.01 * rng.standard_normal(n)
            rel = f"{word}/{spk}_nohash_0.wav"
            sf.write(root / rel, wav.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
            if spk in VAL_SPEAKERS:
                val_lines.append(rel)
            elif spk in TEST_SPEAKERS:
                test_lines.append(rel)
    (root / "validation_list.txt").write_text("\n".join(val_lines) + "\n")
    (root / "testing_list.txt").write_text("\n".join(test_lines) + "\n")

    bg = root / "_background_noise_"
    bg.mkdir()
    for name in ["white_noise", "hum"]:
        noise = 0.1 * rng.standard_normal(3 * SAMPLE_RATE)
        sf.write(bg / f"{name}.wav", noise.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    return root
