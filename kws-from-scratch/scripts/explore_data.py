"""M0 · 先“听”和“看”数据，再写任何模型代码

运行方式：uv run python scripts/explore_data.py

要做的事（每一项都把观察写进学习日志）：
1. 随机挑 5 个词、每个词 3 条录音，用 soundfile 读出来：
   打印 shape、dtype、采样率、时长、最大绝对值。有不到 1 秒的吗？占多少比例？
2. 把其中一条画成三张图（matplotlib，存到 runs/explore/ 下）：
   - 波形（横轴用秒，不是采样点下标）
   - 功率谱图（可以先用 torch.stft，M3 再自己写）
   - log-Mel 图（可以先用 torchaudio.transforms.MelSpectrogram）
   你能从图上看出这个词有几个音节吗？
3. 用 sounddevice.play 亲耳听 10 条。有没有你觉得标注错了或者根本听不清的？
4. 统计每个词的样本数，画柱状图。哪些词最多、最少？差多少倍？这对 M4 的采样器设计意味着什么？
5. 看 _background_noise_ 里每个文件的波形和时长。哪一个最不像“背景噪声”？

这一步没有测试。它的价值在于：之后每次看到某个 shape 或某个数值，你都知道它在真实数据里对应什么。
"""


def main() -> None:
    raise NotImplementedError("M0: explore_data")


if __name__ == "__main__":
    main()
