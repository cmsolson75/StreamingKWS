"""M9 · 评估命令行工具

运行方式
--------
    uv run python -m kws.evaluate --run runs/<run 目录> --split test

没有单元测试。验收标准（写进学习日志）：
- 读 run 目录里的 config.json 和 best 存档，在指定切分上评估
- 输出总体准确率、每类精确率 / 召回率 / F1
- 存一张混淆矩阵图（matplotlib）和一个 report.json 到 run 目录
- 把至少 50 条分错的样本路径写进一个文件，**亲耳去听**，按原因归类
  （说得太快？被噪声盖住？标注本身错了？），这一步比任何指标都重要

思考题
------
1. 评估应该用 EMA 模型还是当前模型？两个都评一下，差多少？
2. 评估时要不要用和训练一样的 SpecAugment / 增强？
3. 存档里的标签顺序和现在 build_labels 的结果不一样（比如你改了关键词列表），会发生什么？怎么防？
"""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    raise NotImplementedError("M9: evaluate.main")


if __name__ == "__main__":
    main()
