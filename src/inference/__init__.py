# 学习目标：理解一个包怎样对外提供统一入口；此文件在导入 src.inference 时执行。
# 先看 __all__ 中的名字，再沿 import 找到各功能实际写在哪个文件。
# 相对导入 .io 指同目录的 io.py；这里重新导出，方便调用方少写一层路径。
from .io import load_cfg_model_state, load_labels, load_model
from .runner import InferenceRunner
from .stream_app import AudioStream, AppState

# ======================================================================
# 1. 声明公开名字：主要控制 from src.inference import * 导入哪些符号
# ======================================================================
# __all__ 是字符串名称列表，不会创建对象，也不会禁止访问未列出的名字。
__all__ = [
    "AudioStream",
    "AppState",
    "InferenceRunner",
    "load_cfg_model_state",
    "load_labels",
    "load_model",
]

# 预测 / 验证：from src.inference import InferenceRunner 后查看 InferenceRunner.__module__，
# 它属于本文件还是 runner.py？__module__ 会显示这个类最初定义的模块名。
# 自己写：在练习包中先建两个小模块，再用 __init__.py 提供统一导入路径。
