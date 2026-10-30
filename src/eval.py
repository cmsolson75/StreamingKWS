# 学习目标：取出已训练好的存档，用指定数据评估，不重新训练模型。
# 阅读路线：main 总流程 → 定位存档 → 恢复配置和参数 → 选择数据 → evaluate。
# 输入是模型路径和可选覆盖参数；输出是平均损失、准确率及本次评估信息。
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from .configs import Config
from .dataloaders import (
    load_background_loader,
    load_dataloader,
    load_oov_loader,
    load_silence_loader,
    load_speech_cmds,
)
from .model import model_factory
from .seed import seed_everything
from .train_utils import evaluate
from .transforms import DbMelSpec


# ======================================================================
# 1. 读取命令行：选择存档、数据类别和运行设备
# ======================================================================
def get_args() -> tuple[argparse.Namespace, list[str]]:
    # ArgumentParser 声明参数；required=True 表示必须传，choices 限定可选值。
    # parse_known_args 返回 (已声明的参数对象, 剩余文字)，剩余内容用于覆盖配置。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="path to checkpoint dir or pointer json (best.json/latest.json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "oov", "kw", "silence", "background"],
        help="which split to evaluate",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="optional device override: cpu|cuda|mps|auto",
    )
    return parser.parse_known_args()


# ======================================================================
# 2. 统一存档路径：直接目录或 best.json / latest.json 都转成目录
# ======================================================================
def resolve_checkpoint_dir(model_path: str) -> Path:
    # 输入路径字符串，输出含 model.safetensors 的目录 Path。
    # suffix 查看扩展名；parent 取上级路径；指针 JSON 的 path 字段是相对目录名。
    path = Path(model_path)
    if path.suffix == ".json":
        pointer = json.loads(path.read_text())
        path = path.parent / pointer["path"]

    # 为什么先检查：路径或参数文件不完整时，尽早报出具体原因。
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"model.safetensors missing in: {path}")

    return path


# ======================================================================
# 3. 从同一次实验中读取配置和权重，保证创建的网络结构与参数匹配
# ======================================================================
def load_cfg_model_state(
    model_path: str,
    device: str | None = None,
    overrides: list[str] | None = None,
) -> tuple[Config, dict[str, torch.Tensor], Path]:
    ckpt_dir = resolve_checkpoint_dir(model_path)
    base_path = ckpt_dir.parent.parent  # 实验目录 / checkpoints / step_xxxxxx，向上两级回到实验目录。
    json_config = base_path / "config.resolved.json"
    if not json_config.exists():
        raise FileNotFoundError(f"config.resolved.json missing at: {json_config}")

    # 先复制覆盖列表；设备参数最后追加，所以同一字段之前的覆盖会被它取代。
    merged_overrides = list(overrides or [])
    if device is not None:
        merged_overrides.append(f"train.device={device}")

    cfg = Config.from_json(str(json_config)).with_overrides(merged_overrides)
    state_dict = load_file(ckpt_dir / "model.safetensors")  # safetensors 读出“参数名 → Tensor”字典。
    return cfg, state_dict, ckpt_dir


def load_model(cfg: Config, state_dict: dict[str, torch.Tensor]) -> nn.Module:
    # 两步缺一不可：model_factory 建结构，load_state_dict 把学到的数值装进去。
    num_classes = len(cfg.data.subset) + 2
    model = model_factory(cfg.model.name, num_classes, cfg.train.dropout).to(
        cfg.train.device
    )
    model.load_state_dict(state_dict)
    model.eval()  # 关闭 Dropout 的随机置零，让 BatchNorm 使用评估方式；不负责禁用梯度记录。
    return model


# ======================================================================
# 4. 按名字选择评估数据：混合数据或只检查某一类
# ======================================================================
def load_eval_loader(cfg: Config, split: str):
    # 注意：oov 和 kw 使用 val 文件划分；这里 test / val 的混合加载器默认各抽 1000 次。
    if split == "test":
        return load_dataloader(cfg, "test")
    if split == "val":
        return load_dataloader(cfg, "val")
    if split == "oov":
        return load_oov_loader(cfg, "val")
    if split == "kw":
        return load_speech_cmds(cfg, "val")
    if split == "silence":
        return load_silence_loader(cfg)
    if split == "background":
        return load_background_loader(cfg)
    raise ValueError(f"unsupported split: {split}")


# ======================================================================
# 5. 串起准备工作，再输出评估结果
# ======================================================================
def main() -> None:
    args, overrides = get_args()

    # 计算速度设置，不会加载权重或执行训练。
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg, state_dict, ckpt_dir = load_cfg_model_state(
        args.model, device=args.device, overrides=overrides
    )
    seed_everything(cfg.env.seed)

    model = load_model(cfg, state_dict)
    # 频谱增强明确关闭；Dataset 中的波形增强仍由 cfg.augment.use_augmentations 控制。
    db_mel_spec = DbMelSpec(cfg, augment=False).to(cfg.train.device)
    loader = load_eval_loader(cfg, args.split)

    # evaluate 在 train_utils.py 中实现，会关闭梯度记录并按样本数汇总指标。
    loss, acc = evaluate(model, cfg, loader, db_mel_spec)
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Device: {cfg.train.device} | Split: {args.split}")
    print(f"Loss: {loss:.4f} | Accuracy: {acc:.4f}")


# 学习练习（准备好已有存档后，在单独脚本或调试器中进行）：
# 预测：best.json 与它指向的 step 目录，传给 resolve_checkpoint_dir 会得到同一目录吗？
# 验证：分别打印两次返回值；修改为 --split kw，追踪为什么最终使用的是 val 数据。
# 自己写：按“读训练配置 → 建网络 → 装权重 → eval → 转输入 → 统计预测”复写评估入口。
if __name__ == "__main__":
    main()
