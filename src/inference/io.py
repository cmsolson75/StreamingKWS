# 学习目标：为实时推理读回“网络结构配置、学到的参数、编号对应的标签”。
# 这三部分必须来自匹配的实验；只有参数字典并不知道应创建哪种网络。
# 与 src/eval.py 的读取流程相似，但这里返回的元组只有 (Config, state_dict)。
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file
from torch import nn

from ..configs import Config
from ..model import model_factory


# ======================================================================
# 1. 读标签：模型第 i 个输出，对应标签列表的第 i 个词
# ======================================================================
def load_labels(label_path: str):
    # Path 管理路径；json.load 读取文件并解析，返回训练时保存的类别名称列表。
    path = Path(label_path)
    with path.open("r") as f:
        return json.load(f)


# ======================================================================
# 2. 定位存档目录，再读取训练配置和权重
# ======================================================================
def load_cfg_model_state(
    model_path: str, device: str | None = None
) -> Tuple[Config, Dict[str, torch.Tensor]]:
    path = Path(model_path)
    # 既接受具体目录，也接受 best.json / latest.json 这样的“指针”文件。
    if path.suffix == ".json":
        json_data = json.loads(path.read_text())["path"]
        path = path.parent / json_data
    base_path = path.parent.parent  # 从实验目录下的 checkpoints/step_xxxxxx 向上两级。

    state_dict = load_file(path / "model.safetensors")  # 返回“参数名称 → Tensor”的字典。
    json_config = base_path / "config.resolved.json"
    device_overwrite = None
    if device is not None:
        device_overwrite = [f"train.device={device}"]  # 设备字段在训练配置的 train 子层。
    return Config.from_json(str(json_config)).with_overrides(
        device_overwrite
    ), state_dict


# ======================================================================
# 3. 建网络结构 → 填入已学参数 → 切到评估方式
# ======================================================================
def load_model(cfg: Config, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    # 为什么还要 +2：输出除关键词外，还包括 UNKNOWN 与 SILENCE 两类。
    num_classes = len(cfg.data.subset) + 2
    model = model_factory(cfg.model.name, num_classes, cfg.train.dropout).to(
        cfg.train.device
    )
    model.load_state_dict(state_dict)  # 参数名和数组形状需要与创建的模型匹配。
    model.eval()
    return model


# 预测 / 验证：load_file 得到的字典能直接处理音频吗？与 load_model 的返回类型比较。
# 在单独脚本读取已有存档后，打印 cfg.model.name 和 list(state_dict)[:3]，再创建模型。
# 修改：将 device 指定为 "cpu"，查看返回模型第一个参数的 .device。
# 自己写：先只支持明确的 step 目录，再补 JSON 指针解析和设备覆盖。
