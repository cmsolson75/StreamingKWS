# 学习目标：使用数据集 API 下载数据，而不是自己处理压缩包和下载地址。
# 本文件是独立的准备入口；导入它不会下载，作为入口运行时才执行下面的代码。
import torchaudio
from pathlib import Path


# ======================================================================
# 1. 确定保存目录 → 创建目录 → 让 TorchAudio 准备 Speech Commands 数据集
# ======================================================================
if __name__ == "__main__":
    # Path.cwd() 是启动命令时的当前目录，/ "data" 在其下面拼出 data 路径。
    data_path = Path.cwd() / "data"
    data_path.mkdir(exist_ok=True)  # 已有该目录时不报错。
    # SPEECHCOMMANDS(root=..., download=True) 创建数据集对象，缺少数据时下载并解压。
    # 后续 dataset.py 会使用配置里的路径寻找这些文件。
    torchaudio.datasets.SPEECHCOMMANDS(root=data_path, download=True)


# 预测 / 验证：先在 Python 中只打印 Path.cwd() / "data"，检查从不同目录启动会保存到哪里。
# 修改 / 自己写：在单独练习文件中指定一个明确的下载根目录，再对齐 cfg.data.path。
