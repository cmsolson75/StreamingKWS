"""M1 · JSONL 指标记录

目标
----
训练时每隔一段时间记一行 JSON（JSON Lines 格式：一行一个 JSON 对象）。
之后画曲线、比较实验，都从这个文件读。

为什么用 JSONL 而不是一个大 JSON 数组
------------------------------------
想想训练进程在第 5000 步被 kill 掉的情况：两种格式的文件分别还能读吗？

先读文档
--------
- JSON Lines：https://jsonlines.org/
- `open` 的模式参数（"a" 和 "w" 的区别）、`file.flush`
- 上下文管理器：`__enter__` / `__exit__`

思考题
------
1. `json.dumps(torch.tensor(0.5))` 会怎样？`json.dumps(np.float32(0.5))` 呢？
   训练循环里 loss 往往是 tensor，你的 logger 该怎么处理？
2. 为什么每写一行要 flush？不 flush 会丢什么？
3. 每次 log 都 open/close 文件，和一直开着文件，各有什么优缺点？
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Self


class JSONLLogger:
    """用法：

        with JSONLLogger(run_dir / "metrics.jsonl") as logger:
            logger.log({"step": 100, "loss": loss})

    要求：
    - 追加写入，不覆盖已有内容（恢复训练时会接着写）
    - 值如果是 0 维 tensor 或 numpy 标量，自动转成 Python 的 float / int
    - 每写一行就 flush
    """

    def __init__(self, path: str | Path) -> None:
        raise NotImplementedError("M1: JSONLLogger.__init__")

    def log(self, record: dict[str, Any]) -> None:
        raise NotImplementedError("M1: JSONLLogger.log")

    def close(self) -> None:
        raise NotImplementedError("M1: JSONLLogger.close")

    def __enter__(self) -> Self:
        raise NotImplementedError("M1: JSONLLogger.__enter__")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError("M1: JSONLLogger.__exit__")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读回所有记录。空行要跳过。"""
    raise NotImplementedError("M1: read_jsonl")
