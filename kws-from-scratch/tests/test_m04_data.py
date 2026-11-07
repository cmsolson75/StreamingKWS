"""M4 验收测试：数据集、采样器、collate、DataLoader。不要修改。

用的是 conftest.py 里造的迷你假数据集。
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader

from kws.collate import Batch, lengths_to_mask, pad_collate
from kws.config import Config
from kws.dataloaders import build_dataloader
from kws.dataset import (
    SILENCE,
    UNKNOWN,
    BackgroundDataset,
    KeywordDataset,
    SilenceDataset,
    UnknownDataset,
    build_labels,
    list_split_files,
)
from kws.sampler import GroupWeightedSampler

from .conftest import FAKE_KEYWORDS

N = 16000


# ---------------------------------------------------------------- labels / split
def test_build_labels():
    assert build_labels(["b", "a"]) == ["b", "a", UNKNOWN, SILENCE]


def test_split_files_are_disjoint_and_complete(fake_speech_commands):
    splits = {s: list_split_files(fake_speech_commands, s) for s in ["train", "val", "test"]}
    assert len(splits["val"]) == 6 and len(splits["test"]) == 6 and len(splits["train"]) == 24
    sets = {k: set(v) for k, v in splits.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"])
    assert all(p.is_absolute() for p in splits["train"])
    assert all("_background_noise_" not in str(p) for p in splits["train"])
    assert all("spk4" in p.name for p in splits["val"])
    assert splits["train"] == sorted(splits["train"])


# ---------------------------------------------------------------- datasets
def test_keyword_dataset(fake_speech_commands):
    labels = build_labels(FAKE_KEYWORDS)
    ds = KeywordDataset(fake_speech_commands, "train", FAKE_KEYWORDS, labels, N)
    assert len(ds) == 4 * 4  # 4 个关键词 × 4 个训练说话人
    seen = Counter()
    for i in range(len(ds)):
        wav, y = ds[i]
        assert wav.shape == (N,) and wav.dtype == torch.float32
        assert isinstance(y, int)
        seen[labels[y]] += 1
    assert set(seen) == set(FAKE_KEYWORDS)


def test_short_clips_are_padded(fake_speech_commands):
    labels = build_labels(FAKE_KEYWORDS)
    ds = KeywordDataset(fake_speech_commands, "train", FAKE_KEYWORDS, labels, N)
    shapes = {ds[i][0].shape for i in range(len(ds))}
    assert shapes == {(N,)}


def test_unknown_dataset(fake_speech_commands):
    labels = build_labels(FAKE_KEYWORDS)
    ds = UnknownDataset(fake_speech_commands, "val", FAKE_KEYWORDS, labels, N)
    assert len(ds) == 2  # bed、cat 各 1 个验证说话人
    assert {ds[i][1] for i in range(len(ds))} == {labels.index(UNKNOWN)}


def test_transform_is_applied(fake_speech_commands):
    labels = build_labels(FAKE_KEYWORDS)
    ds = KeywordDataset(
        fake_speech_commands, "test", FAKE_KEYWORDS, labels, N, transform=lambda w: w * 0
    )
    assert ds[0][0].abs().sum() == 0


def test_silence_dataset_is_quiet(fake_speech_commands):
    labels = build_labels(FAKE_KEYWORDS)
    ds = SilenceDataset(labels, N, length=50, noise_floor_db=-60.0)
    assert len(ds) == 50
    wav, y = ds[3]
    assert wav.shape == (N,) and labels[y] == SILENCE
    rms_db = 10 * torch.log10(wav.pow(2).mean() + 1e-12)
    assert rms_db <= -59.0


def test_background_dataset(fake_speech_commands):
    labels = build_labels(FAKE_KEYWORDS)
    ds = BackgroundDataset(fake_speech_commands, labels, N, length=20)
    assert len(ds) == 20
    a, y = ds[0]
    assert a.shape == (N,) and labels[y] == SILENCE
    assert a.abs().sum() > 0


# ---------------------------------------------------------------- sampler
def test_sampler_basic_properties():
    sizes = [100, 1000, 10, 10]
    s = GroupWeightedSampler(
        sizes, [0.7, 0.2, 0.05, 0.05], num_samples=500, generator=torch.Generator().manual_seed(0)
    )
    idx = list(s)
    assert len(s) == 500 and len(idx) == 500
    assert all(isinstance(i, int) and 0 <= i < sum(sizes) for i in idx)


def test_sampler_follows_group_weights():
    sizes = [100, 1000, 10, 10]
    weights = [7, 2, 0.5, 0.5]  # 不需要事先归一化
    s = GroupWeightedSampler(
        sizes, weights, num_samples=20000, generator=torch.Generator().manual_seed(0)
    )
    bounds = torch.tensor(sizes).cumsum(0).tolist()
    groups = Counter(next(g for g, b in enumerate(bounds) if i < b) for i in s)
    for g, expected in enumerate([0.7, 0.2, 0.05, 0.05]):
        assert groups[g] / 20000 == pytest.approx(expected, abs=0.015)


def test_sampler_is_reproducible():
    def make(seed):
        return GroupWeightedSampler(
            [5, 5], [1, 1], 50, generator=torch.Generator().manual_seed(seed)
        )

    assert list(make(1)) == list(make(1))
    assert list(make(1)) != list(make(2))


def test_sampler_state_dict_resumes_mid_stream():
    full = list(
        GroupWeightedSampler([30, 70], [1, 1], 100, generator=torch.Generator().manual_seed(0))
    )

    s = GroupWeightedSampler([30, 70], [1, 1], 100, generator=torch.Generator().manual_seed(0))
    it = iter(s)
    first = [next(it) for _ in range(40)]
    state = s.state_dict()

    resumed = GroupWeightedSampler(
        [30, 70], [1, 1], 100, generator=torch.Generator().manual_seed(999)
    )
    resumed.load_state_dict(state)
    rest = list(resumed)[:60]
    assert first == full[:40]
    assert rest == full[40:]


def test_sampler_works_with_concat_dataset(fake_speech_commands):
    labels = build_labels(FAKE_KEYWORDS)
    kw = KeywordDataset(fake_speech_commands, "train", FAKE_KEYWORDS, labels, N)
    unk = UnknownDataset(fake_speech_commands, "train", FAKE_KEYWORDS, labels, N)
    concat = ConcatDataset([kw, unk])
    s = GroupWeightedSampler(
        [len(kw), len(unk)], [0.0, 1.0], 20, generator=torch.Generator().manual_seed(0)
    )
    ys = {concat[i][1] for i in s}
    assert ys == {labels.index(UNKNOWN)}


# ---------------------------------------------------------------- collate
def test_lengths_to_mask():
    m = lengths_to_mask(torch.tensor([3, 1, 0]), max_len=4)
    assert m.dtype == torch.bool
    assert m.tolist() == [[True, True, True, False], [True, False, False, False], [False] * 4]
    assert lengths_to_mask(torch.tensor([2, 5])).shape == (2, 5)


def test_pad_collate_variable_lengths():
    samples = [(torch.ones(3), 1), (torch.full((5,), 2.0), 0), (torch.ones(1), 2)]
    b = pad_collate(samples)
    assert isinstance(b, Batch)
    assert b.waveforms.shape == (3, 5)
    assert b.lengths.tolist() == [3, 5, 1] and b.lengths.dtype == torch.int64
    assert b.labels.tolist() == [1, 0, 2] and b.labels.dtype == torch.int64
    assert b.waveforms[0].tolist() == [1, 1, 1, 0, 0]
    assert torch.equal(b.mask, lengths_to_mask(b.lengths, 5))
    assert samples[0][0].shape == (3,), "不能修改输入"


def test_pad_collate_in_dataloader():
    data = [(torch.randn(n), n % 3) for n in [10, 7, 12, 3]]
    batch = next(iter(DataLoader(data, batch_size=4, collate_fn=pad_collate)))
    assert batch.waveforms.shape == (4, 12)


# ---------------------------------------------------------------- dataloader
def _cfg(config_path, root):
    return Config.from_yaml(config_path).with_overrides(
        [
            f"data.root={root}",
            f"data.keywords=[{','.join(FAKE_KEYWORDS)}]",
            "train.batch_size=8",
            "train.num_workers=0",
            "train.max_steps=5",
            "train.warmup_steps=0",
            "augment.enabled=false",
        ]
    )


def test_train_loader(config_path, fake_speech_commands):
    cfg = _cfg(config_path, fake_speech_commands)
    labels = build_labels(cfg.data.keywords)
    loader = build_dataloader(cfg, "train", labels)
    batches = list(loader)
    assert len(batches) == 5
    x, y = batches[0]
    assert x.shape == (8, N) and y.shape == (8,)
    assert int(y.max()) < len(labels)


def test_eval_loader_is_deterministic(config_path, fake_speech_commands):
    cfg = _cfg(config_path, fake_speech_commands)
    labels = build_labels(cfg.data.keywords)
    a = torch.cat([y for _, y in build_dataloader(cfg, "val", labels)])
    b = torch.cat([y for _, y in build_dataloader(cfg, "val", labels)])
    assert torch.equal(a, b)
    # 6 条真实验证录音必须全部出现
    assert (a < len(labels) - 1).sum() >= 6
