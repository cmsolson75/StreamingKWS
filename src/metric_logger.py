# 学习目标：把每次训练指标追加到文件，保留整个训练过程，而不是只保留最后一次。
# 输入是一条字典记录，输出是 metrics.jsonl 中新增的一行 JSON 文本。
from pathlib import Path
import json


# ======================================================================
# 1. 先记住日志位置，再按行追加记录
# ======================================================================
class JSONLMetricLogger:
    def __init__(self, path: Path):
        self.jsonl_path = path / "metrics.jsonl"  # Path 用 / 拼接文件名；上级目录由调用方准备。

    def log(self, data: dict) -> None:
        # mode="a" 表示追加；"w" 会覆盖旧内容；with 结束时自动关闭文件。
        with self.jsonl_path.open(mode="a", encoding="utf-8") as f:
            json_line = json.dumps(data)  # 字典转成 JSON 字符串；数据需先转成普通可序列化数值。
            f.write(json_line + "\n")


# 预测 / 验证：在临时目录创建 logger，连续 log({"step": 1})、log({"step": 2})，应有几行？
# 修改：在单独练习文件中比较追加 a 与覆盖 w，理解为什么训练日志选择 a。
# 自己写：按“指定路径 → 字典转 JSON → 追加一行”复写，不必先接入模型训练。
