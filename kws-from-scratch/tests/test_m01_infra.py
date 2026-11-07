"""M1 验收测试：配置、seed、provenance、日志。不要修改。"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from kws.config import Config
from kws.logger import JSONLLogger, read_jsonl
from kws.provenance import create_run_dir, get_provenance
from kws.seed import make_generator, seed_everything


# ----------------------------------------------------------------- config
def test_config_loads_yaml(config_path):
    cfg = Config.from_yaml(config_path)
    assert cfg.preprocess.n_mels == 64
    assert cfg.preprocess.num_samples == 16000
    assert cfg.data.keywords[0] == "marvin"
    assert cfg.model.name == "tc_resnet8"
    assert cfg.env.resume is None


def test_config_is_frozen(config_path):
    cfg = Config.from_yaml(config_path)
    with pytest.raises(ValidationError):
        cfg.train.lr = 1.0  # type: ignore[misc]


def test_config_rejects_unknown_fields(config_path):
    raw = Config.from_yaml(config_path).model_dump()
    raw["train"]["learning_rate_typo"] = 0.1
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_overrides_parse_types_and_return_new_object(config_path):
    cfg = Config.from_yaml(config_path)
    new = cfg.with_overrides(
        [
            "train.batch_size=32",
            "augment.enabled=false",
            "train.lr=3e-4",
            "env.resume=null",
            "data.keywords=[marvin,stop]",
        ]
    )
    assert new is not cfg
    assert new.train.batch_size == 32 and isinstance(new.train.batch_size, int)
    assert new.augment.enabled is False
    assert new.train.lr == pytest.approx(3e-4)
    assert new.data.keywords == ["marvin", "stop"]
    # 原对象不变
    assert cfg.train.batch_size == 64
    assert cfg.augment.enabled is True


def test_overrides_unknown_key_raises(config_path):
    cfg = Config.from_yaml(config_path)
    with pytest.raises((KeyError, ValueError)):
        cfg.with_overrides(["train.not_a_field=1"])
    with pytest.raises((KeyError, ValueError)):
        cfg.with_overrides(["nope.batch_size=1"])


def test_overrides_bad_format_raises(config_path):
    cfg = Config.from_yaml(config_path)
    with pytest.raises(ValueError):
        cfg.with_overrides(["train.batch_size"])


def test_validators_run_after_override(config_path):
    cfg = Config.from_yaml(config_path)
    with pytest.raises(ValidationError):
        cfg.with_overrides(["sampler.keyword=0.9"])  # 权重总和不再是 1
    with pytest.raises(ValidationError):
        cfg.with_overrides(["augment.min_snr_db=30"])  # 下限 > 上限
    with pytest.raises(ValidationError):
        cfg.with_overrides(["augment.noise_prob=1.5"])
    with pytest.raises(ValidationError):
        cfg.with_overrides(["model.name=resnet50"])
    with pytest.raises(ValidationError):
        cfg.with_overrides(["train.warmup_steps=999999"])


def test_hash_is_stable_and_sensitive(config_path):
    a = Config.from_yaml(config_path)
    b = Config.from_yaml(config_path)
    assert a.hash() == b.hash()
    assert len(a.hash()) == 10 and len(a.hash(16)) == 16
    int(a.hash(), 16)  # 必须是十六进制
    assert a.with_overrides(["train.lr=0.002"]).hash() != a.hash()


def test_json_round_trip(config_path, tmp_path):
    cfg = Config.from_yaml(config_path).with_overrides(["train.batch_size=8"])
    cfg.to_json(tmp_path / "c.json")
    assert Config.from_json(tmp_path / "c.json") == cfg
    json.loads((tmp_path / "c.json").read_text())  # 是合法 JSON


def test_resolved_device(config_path):
    cfg = Config.from_yaml(config_path)
    assert cfg.resolved_device() in {"cpu", "cuda", "mps"}
    assert cfg.with_overrides(["train.device=cpu"]).resolved_device() == "cpu"


# ------------------------------------------------------------------- seed
def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def test_seed_everything_controls_all_sources():
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    b = _draw()
    seed_everything(124)
    c = _draw()
    assert a == b
    assert all(x != y for x, y in zip(a, c))


def test_make_generator_is_independent_of_global_state():
    g1 = make_generator(7)
    x = torch.rand(3, generator=g1)
    torch.manual_seed(999)  # 改全局种子不应影响独立 generator
    g2 = make_generator(7)
    assert torch.equal(x, torch.rand(3, generator=g2))


# ------------------------------------------------------------- provenance
def test_provenance_outside_git_does_not_crash(tmp_path):
    info = get_provenance(tmp_path)
    for key in [
        "git_commit",
        "git_dirty",
        "python",
        "torch",
        "platform",
        "hostname",
        "time_utc",
        "cuda_available",
    ]:
        assert key in info, key
    assert info["git_commit"] is None
    assert info["torch"] == torch.__version__
    json.dumps(info)  # 必须能序列化


def test_create_run_dir(config_path, tmp_path):
    cfg = Config.from_yaml(config_path)
    run = create_run_dir(cfg, root=tmp_path)
    assert run.parent == tmp_path
    assert run.name.endswith("_" + cfg.hash())
    assert (run / "checkpoints").is_dir()
    assert Config.from_json(run / "config.json") == cfg
    assert "git_commit" in json.loads((run / "provenance.json").read_text())


# ----------------------------------------------------------------- logger
def test_jsonl_logger_appends_and_converts(tmp_path):
    path = tmp_path / "m.jsonl"
    with JSONLLogger(path) as log:
        log.log({"step": 1, "loss": torch.tensor(0.5), "acc": np.float32(0.25)})
        log.log({"step": 2, "loss": 0.4})
    with JSONLLogger(path) as log:  # 再次打开：追加，不覆盖
        log.log({"step": 3, "n": np.int64(7)})
    rows = read_jsonl(path)
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert rows[0]["loss"] == pytest.approx(0.5) and isinstance(rows[0]["loss"], float)
    assert rows[0]["acc"] == pytest.approx(0.25)
    assert rows[2]["n"] == 7


def test_jsonl_logger_flushes_each_line(tmp_path):
    path = tmp_path / "m.jsonl"
    log = JSONLLogger(path)
    log.log({"step": 1})
    assert read_jsonl(path) == [{"step": 1}]  # 还没 close，也必须已经写到磁盘
    log.close()
