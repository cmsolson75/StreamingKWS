"""M10 · 环形缓冲区

背景
----
麦克风回调函数在**另一个线程**里每隔几十毫秒送来一小块音频；
推理循环在主线程里每 0.1 秒取“最近 1 秒”的音频去识别。
两个线程同时读写同一块内存，需要一个固定大小、线程安全的环形缓冲区。

先读文档
--------
- threading.Lock（以及 with lock: 的写法）
- numpy 的切片和取模下标

思考题
------
1. 不加锁会出什么问题？构造一个具体的交错执行顺序，说明 read() 可能读到“一半新、一半旧”的数据。
2. 一次写入的数据比缓冲区还长，应该保留哪一部分？
3. 读出来的数组如果直接返回内部数组的切片（不复制），会有什么隐患？
4. 锁里面应该只做什么？为什么不要在持有锁时做推理？
"""

from __future__ import annotations

import numpy as np


class CircularBuffer:
    """容量固定的一维环形缓冲区。

    - write(chunk)：追加任意长度的一维数组（可以比容量长）
    - read()：按时间顺序返回最近 capacity 个采样点，返回副本；还没写满时，前面用 0 补
    - total_written：累计写入的采样点数
    - 所有方法线程安全
    """

    def __init__(self, capacity: int, dtype: np.dtype | type = np.float32) -> None:
        raise NotImplementedError("M10: CircularBuffer.__init__")

    def write(self, chunk: np.ndarray) -> None:
        raise NotImplementedError("M10: CircularBuffer.write")

    def read(self) -> np.ndarray:
        raise NotImplementedError("M10: CircularBuffer.read")

    @property
    def total_written(self) -> int:
        raise NotImplementedError("M10: CircularBuffer.total_written")
