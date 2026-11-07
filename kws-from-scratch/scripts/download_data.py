"""M0 · 下载 Speech Commands v0.02

运行方式：uv run python scripts/download_data.py

目标：把数据集下载并解压到 data/SpeechCommands/speech_commands_v0.02/

两种做法，任选一种（都试一下更好）：
1. torchaudio.datasets.SPEECHCOMMANDS(root=..., url="speech_commands_v0.02", download=True)
   只用它来下载，后面不用它的 Dataset 类，Dataset 你自己在 M4 写。
2. 自己下载 tar.gz（地址在 Speech Commands 论文和 TensorFlow Datasets 文档里都能找到），
   用 urllib.request + tarfile 解压。思考：tarfile.extractall 有什么安全隐患？filter 参数是做什么的？

下载完后检查：
- 有多少个词目录？每个目录多少个文件？
- _background_noise_ 里有哪些文件，各多长？
- validation_list.txt、testing_list.txt 各多少行？
把这些数字写进学习日志。
"""


def main() -> None:
    raise NotImplementedError("M0: download_data")


if __name__ == "__main__":
    main()
