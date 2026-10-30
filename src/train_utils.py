# 学习目标：找到模型真正“学习”的地方，而不只是创建网络和优化器。
# 阅读顺序：training_step（学一批）→ train（重复、验证、存档）→ evaluate（只检查）。
# 一次学习：取数据 → 预测 → 计算误差 → 求梯度 → 更新参数；梯度提供参数改动的信息。
import torch
from torch import nn
import torch.nn.functional as F
from ema_pytorch import EMA

from .configs import Config
from collections.abc import Iterator
from .transforms import DbMelSpec
from typing import Tuple
from .checkpoint_manager import CheckpointManager
from .metric_logger import JSONLMetricLogger


# ======================================================================
# 1. 学一批：从声音和正确标签，得到损失，并更新模型
# ======================================================================
def training_step(
    model: nn.Module,
    ema_model: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    iter_loader: Iterator[Tuple[torch.Tensor, int]],
    cfg: Config,
    db_mel_spec: DbMelSpec,
):
    # ---------- 1.1 取下一批数据，转换成模型的输入 ----------
    # train() 切到训练方式，例如启用 Dropout；它本身不执行一次训练。
    model.train()
    # next 从已有迭代器拿下一批；不要每步重新 iter(loader)，否则可能反复从头取。
    x, y = next(iter_loader)
    x, y = x.to(cfg.train.device), y.to(cfg.train.device)
    # .to(...) 将数据放到与模型相同的设备；频谱转换 [B, 1, N] → [B, 1, F, T]。
    x = db_mel_spec(x)

    # ---------- 1.2 模型预测，并用正确标签计算误差 ----------
    # autocast 按操作选择合适的计算精度；enabled=False 时不启用混合精度。
    with torch.autocast(device_type=cfg.train.device, enabled=cfg.train.amp):
        logits = model(x)
        # cross_entropy 接收原始分数 [B, 类别数] 和整数标签 [B]，返回一个平均损失。
        # 不要先手动 softmax；这个 API 已包含相应的对数概率计算。
        loss = F.cross_entropy(logits, y)

    # ---------- 1.3 求每个参数的梯度，限制过大更新，再让优化器改参数 ----------
    optimizer.zero_grad()  # 清掉上一次梯度，否则本次梯度默认会继续累加。

    # scale 放大损失以减少小数值丢失；backward 根据误差计算各参数的梯度。
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # 先还原梯度尺度，再裁剪；clip_grad_norm_ 限制总梯度大小，返回裁剪前的总范数。
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_norm)

    # scaler.step 间接调用优化器更新；检测到无效梯度时可以跳过本次更新。
    scaler.step(optimizer)
    ema_model.update()  # 更新模型参数的平滑平均版本，用于验证。
    scaler.update()  # 根据数值情况调整下一次使用的缩放倍率。
    scheduler.step()  # 将学习率计划推进一步。
    return loss, total_norm


# ======================================================================
# 2. 重复训练：学一批 → 按间隔记日志 → 验证 → 保存进度
# ======================================================================
def train(
    model: nn.Module,
    ema_model: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: Config,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    eval_loaders: list[tuple[str, torch.utils.data.DataLoader]],
    eval_period: int,
    log_period: int,
    db_mel_spec: DbMelSpec,
    checkpoint_manager: CheckpointManager,
    start_step: int,
    metric_logger: JSONLMetricLogger,
    best_acc: float,
):
    model.train()

    # iter 创建可持续取样的迭代器；range 的右端不包含，因此这里会包含 max_steps 这一步。
    it = iter(train_loader)
    for step in range(start_step, cfg.train.max_steps + 1):
        train_loss, total_norm = training_step(
            model, ema_model, optimizer, scaler, scheduler, it, cfg, db_mel_spec
        )

        # % 是取余；例如 step % 200 == 0 表示到了记录间隔，step=0 也满足。
        if step % log_period == 0:
            print(
                f"{step}/{cfg.train.max_steps}: train_loss={train_loss.item():.4f}, norm={total_norm:.4f}"
            )
            metric_logger.log(
                {
                    "split": "train",
                    "loss": train_loss.item(),
                    "step": step,
                    "total_norm": total_norm.item(),
                }
            )
        # 验证用 EMA 的平均模型，不是本轮刚更新的原模型。
        if step % eval_period == 0:
            val_loss, val_acc = evaluate(
                ema_model.ema_model, cfg, val_loader, db_mel_spec
            )

            output_metrics = {"val_loss": val_loss, "val_acc": val_acc}
            # 分别统计各类数据，避免一个总准确率掩盖某一类的差表现。
            for name, loader in eval_loaders:
                _, eval_acc = evaluate(ema_model.ema_model, cfg, loader, db_mel_spec)
                output_metrics[f"{name}_acc"] = eval_acc

            parts = [
                f"{step}/{cfg.train.max_steps}",
            ]
            for k, v in output_metrics.items():
                parts.append(f"{k}={v:.4f}")
            print("  ".join(parts))

            # **字典 将其中每个键值合入新字典；.item() 则把单个 Tensor 数值取出来用于日志。
            metric_logger.log({"split": "val", "step": step, **output_metrics})
            best = val_acc > best_acc
            if best:
                best_acc = val_acc
            # 保存平均模型参数，以及优化器等状态；best 仅在准确率刷新最高纪录时为 True。
            checkpoint_manager.save(
                ema_model.ema_model,
                step,
                best=best,
                best_acc=best_acc,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                sampler_step=step * cfg.train.batch_size,
            )


# ======================================================================
# 3. 评估：只做预测和统计，不求训练梯度、不更新参数
# ======================================================================
# @no_grad 装饰整个函数，关闭梯度记录；eval() 另外负责改变 Dropout、BatchNorm 的行为。
@torch.no_grad()
def evaluate(model: nn.Module, cfg: Config, loader, db_mel_spec: DbMelSpec):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(cfg.train.device), y.to(cfg.train.device)
        x = db_mel_spec(x)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        bs = x.size(0)  # size(0) 是这一批的实际样本数，最后一批可能比 batch_size 小。
        # 为什么先乘 bs：每批准确率要按样本数加权，不能让小批与大批占同等份量。
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs

    return total_loss / n, total_acc / n


# ======================================================================
# 4. 从类别分数算准确率：选最高分 → 对答案 → 求猜对的比例
# ======================================================================
def accuracy(logits: torch.Tensor, ys: torch.Tensor):
    # argmax(dim=1) 每行选最高分类别；== 得到布尔数组；float() 变成 0/1，mean() 求平均。
    return (logits.argmax(dim=1) == ys).float().mean().item()


# 预测 / 验证（单独脚本里导入 torch 和 accuracy）：
# scores = torch.tensor([[0., 3.], [2., 1.]])，正确标签是 torch.tensor([1, 1])，准确率是多少？
# 执行 accuracy(scores, 正确标签)，再把第二个标签改为 0，比较结果。
# 研究：去 training_step 逐行标出“仅准备”和“确实修改参数”的调用。
# 自己写：先用普通精度复写“清梯度 → 前向 → loss → backward → step”，再加入 scaler 和裁剪。
