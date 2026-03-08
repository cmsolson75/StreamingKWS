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
    model.train()
    x, y = next(iter_loader)
    x, y = x.to(cfg.train.device), y.to(cfg.train.device)
    # Batched
    x = db_mel_spec(x)

    with torch.autocast(device_type=cfg.train.device, enabled=cfg.train.amp):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_norm)

    scaler.step(optimizer)
    ema_model.update()
    scaler.update()
    scheduler.step()
    return loss, total_norm


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

    it = iter(train_loader)
    for step in range(start_step, cfg.train.max_steps + 1):
        train_loss, total_norm = training_step(
            model, ema_model, optimizer, scaler, scheduler, it, cfg, db_mel_spec
        )

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
        if step % eval_period == 0:
            val_loss, val_acc = evaluate(
                ema_model.ema_model, cfg, val_loader, db_mel_spec
            )

            output_metrics = {"val_loss": val_loss, "val_acc": val_acc}
            for name, loader in eval_loaders:
                _, eval_acc = evaluate(ema_model.ema_model, cfg, loader, db_mel_spec)
                output_metrics[f"{name}_acc"] = eval_acc

            parts = [
                f"{step}/{cfg.train.max_steps}",
            ]
            for k, v in output_metrics.items():
                parts.append(f"{k}={v:.4f}")
            print("  ".join(parts))

            metric_logger.log({"split": "val", "step": step, **output_metrics})
            best = val_acc > best_acc
            if best:
                best_acc = val_acc
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
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs

    return total_loss / n, total_acc / n


def accuracy(logits: torch.Tensor, ys: torch.Tensor):
    return (logits.argmax(dim=1) == ys).float().mean().item()
