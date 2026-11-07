"""M10（最后一步）· 麦克风实时应用

运行方式
--------
    uv run python -m kws.streaming.app --run runs/<run 目录>

**前提**：test_m10_streaming.py 全绿，并且你已经用自己录的一段 WAV
（说 “marvin one two stop”）跑过 StreamingDetector.run_on_waveform，事件正确。

没有单元测试。验收标准：
- 对着麦克风说 “marvin … three … five … stop”，打印出 [3, 5]
- 安静地放着 10 分钟，记录误报次数
- 测一下从说完 “stop” 到打印结果的延迟

先读文档
--------
- sounddevice.InputStream：https://python-sounddevice.readthedocs.io/ 重点：callback 的签名、blocksize、
  callback 在哪个线程里运行、callback 里能不能做耗时的事
- 键盘中断（KeyboardInterrupt）时怎么干净地关掉音频流

结构建议（自己决定细节）
----------------------
- callback：只做一件事，把 indata 写进 CircularBuffer
- 主循环：每 hop 秒读一次缓冲区，复用 StreamingDetector 里的推理逻辑（想想怎么拆，才能让离线和在线共用代码）
- 收到事件就打印
"""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    raise NotImplementedError("M10: streaming app")


if __name__ == "__main__":
    main()
