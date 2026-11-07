"""M1 · 实验来源记录（provenance）和 run 目录

目标
----
每次训练都在 `runs/` 下建一个新目录，里面写清楚“这次结果是怎么来的”：
用的配置、代码版本、代码有没有未提交的改动、软件版本、机器信息。

为什么
------
三个月后你看到一个 98% 的结果，想复现它。如果只知道“大概是那周跑的”，就复现不了。
provenance 回答的问题是：要得到同样的结果，我需要恢复哪些东西？

先读文档
--------
- `subprocess.run`：重点看 `capture_output`、`text`、`check`、`cwd`，以及失败时抛什么异常
- `git rev-parse HEAD`、`git status --porcelain` 各输出什么（自己在终端里跑一遍）
- `platform`、`socket.gethostname`、`datetime.now(timezone.utc)`

思考题
------
1. 在一个不是 git 仓库的目录里跑 `git rev-parse HEAD` 会怎样？电脑上根本没装 git 又会怎样？
   这两种情况分别是哪种异常？你的函数应该崩溃吗？
2. 只记 commit 哈希为什么不够？“dirty” 指什么？
3. 为什么 run 目录名里要同时有时间戳和配置哈希？只用其中一个会出什么问题？
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kws.config import Config


def get_provenance(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """收集来源信息，返回一个**能被 json.dumps 直接序列化**的 dict。

    必须包含这些键（值的类型在括号里）：
    - "git_commit"（str 或 None）：repo_dir 不是 git 仓库、或者没装 git 时为 None，不能抛异常
    - "git_dirty"（bool 或 None）：有未提交改动为 True；拿不到信息时为 None
    - "python"（str）、"torch"（str）：版本号
    - "platform"（str）、"hostname"（str）
    - "time_utc"（str）：ISO 8601 格式
    - "cuda_available"（bool）

    repo_dir 为 None 时用当前目录。
    """
    raise NotImplementedError("M1: get_provenance")


def create_run_dir(cfg: Config, root: str | Path | None = None) -> Path:
    """在 root（默认 cfg.env.run_root）下新建 run 目录并写入两份记录，返回目录路径。

    要求：
    - 目录名形如 "20261106T153000Z_<cfg.hash()>"，即 UTC 时间戳 + 下划线 + 配置哈希
    - 目录里有 "config.json"（cfg.to_json 写出）和 "provenance.json"
    - 同时建好子目录 "checkpoints"
    - 目录已存在时报错，绝不能覆盖旧实验（思考：什么情况下会撞名？）
    """
    raise NotImplementedError("M1: create_run_dir")
