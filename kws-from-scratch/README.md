# kws-from-scratch：闭卷从零写一个流式关键词识别系统

这是一个**填空式**学习项目。目录结构、函数签名、说明和验收测试都已经给你，但**函数体全是空的**（`raise NotImplementedError`）。你的任务是把它们一个个填完，最后得到一个和 StreamingKWS 同等能力的系统：

> 麦克风音频 → Mel 频谱 → TC-ResNet8 → 平滑 + 状态机 → 识别出 “marvin … 一串数字 … stop”

它和 [学习计划](../docs/noel/学习计划.md) 配套。计划讲“学什么”，这里是“在哪写”。

---

## 规则（先读，再开始）

1. **闭卷写**：写 `kws/` 里的代码时，不打开仓库根目录的 `src/`（StreamingKWS 原代码），不让 AI 写代码，关掉编辑器的 AI 补全。可以查官方文档、论文、自己的笔记。
2. **测试不能改**：`tests/test_mXX_*.py` 是验收标准。测试不过就改你的代码，不改测试。觉得测试本身有问题，先写下理由再问。
3. **自己也写测试**：每个里程碑至少在 `tests/mine/` 加 3 个你自己的测试。给你的测试只覆盖“必须做到的”，没覆盖边界情况。
4. **先预测，再运行**：每个文件顶部都有“思考题”。写代码前先把答案写进 [docs/学习日志.md](docs/学习日志.md)，跑完再对照。
5. **提示分级**：卡住 30 分钟以上再打开 `hints/` 里对应文件，**一次只展开一级**。用了几级提示也记进日志，这是你下次复习的重点。

---

## 每个里程碑怎么做

```text
读文件顶部说明 → 回答思考题 → 填函数 → uv run pytest tests/test_mXX_*.py
→ 自己加测试 → 隔天删掉最核心的函数，闭卷重写一遍 → 在下面打勾
```

运行环境：

```bash
uv sync                    # 装依赖
uv sync --extra export     # M11 需要 onnx / onnxruntime
uv run pytest tests/test_m01_infra.py   # 只跑某个里程碑
uv run pytest              # 全部（一开始几乎全红，正常）
```

---

## 里程碑地图

顺序有讲究：后面的模块会 import 前面的。**测试全绿再进入下一个。**

| # | 主题 | 你要填的文件 | 验收测试 | 计划阶段 |
|---|---|---|---|---|
| M0 | 环境、数据、看懂音频 | `scripts/download_data.py`、`scripts/explore_data.py` | 无（交一张图和笔记） | 0 |
| M1 | 实验基础设施 | `kws/config.py`、`kws/seed.py`、`kws/provenance.py`、`kws/logger.py` | `test_m01_infra.py` | 1 |
| M2 | 音频 I/O | `kws/audio.py` | `test_m02_audio.py` | 3 |
| M3 | 特征：STFT → Mel → dB | `kws/features.py` | `test_m03_features.py` | 3 |
| M4 | 数据集、采样、collate | `kws/dataset.py`、`kws/sampler.py`、`kws/collate.py` | `test_m04_data.py` | 3 |
| M5 | 数据增强 | `kws/augment.py` | `test_m05_augment.py` | 3 |
| M6 | 模型 | `kws/model.py` | `test_m06_model.py` | 4 |
| M7 | 优化器、学习率、EMA、训练循环 | `kws/optim.py`、`kws/schedule.py`、`kws/ema.py`、`kws/train.py` | `test_m07_training.py` | 2、4 |
| M8 | 存档与严格恢复 | `kws/checkpoint.py` | `test_m08_checkpoint.py` | 4 |
| M9 | 评估指标 | `kws/metrics.py`、`kws/evaluate.py` | `test_m09_metrics.py` | 5 |
| M10 | 流式推理 | `kws/streaming/buffer.py`、`kws/streaming/detector.py`、`kws/streaming/app.py` | `test_m10_streaming.py` | 6 |
| M11 | 导出 ONNX | `kws/export.py` | `test_m11_export.py` | 6 |
| M12 | Transformer（选做） | `kws/transformer.py` | `test_m12_transformer.py` | 7 |

M8 的“严格恢复测试”和 M10 的“状态机测试”是整个项目最难也最值钱的两关。

**毕业考试不在这里**：计划阶段 8 要求你在一个**新的空 repo** 里，不靠任何骨架做 ESC-50。这个骨架帮你学会“怎么拆”，毕业考试检验你能不能自己拆。

---

## 目录

```text
kws-from-scratch/
├── configs/config.yaml     # 给你的配置数据，schema 由你写
├── kws/                    # 你要填的代码
│   └── streaming/
├── scripts/                # 下载数据、探索数据、训练入口
├── tests/
│   ├── conftest.py         # 造假数据的 fixture（看懂它，也是练习）
│   ├── test_mXX_*.py       # 验收测试，不要改
│   └── mine/               # 你自己写的测试
├── hints/                  # 分级提示，卡住再看
└── docs/学习日志.md         # 预测、结果、错误、用了几级提示
```

---

## 进度

只有“测试全绿 + 隔天闭卷重写过核心函数”才打勾。

- [ ] M0 环境与数据
- [ ] M1 实验基础设施
- [ ] M2 音频 I/O
- [ ] M3 特征
- [ ] M4 数据集、采样、collate
- [ ] M5 数据增强
- [ ] M6 模型
- [ ] M7 训练循环（并在真实数据上训练出 > 90% 验证准确率）
- [ ] M8 严格恢复
- [ ] M9 评估报告
- [ ] M10 流式推理（离线 WAV 测试通过，再接麦克风）
- [ ] M11 ONNX 导出
- [ ] M12 Transformer（选做）
