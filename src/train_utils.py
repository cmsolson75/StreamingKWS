import torch
from torch import nn
import torch.nn.functional as F
from ema_pytorch import EMA

from .dataloaders import infinite_dataloader
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
    x, y = x.to(cfg.device), y.to(cfg.device)
    # Batched
    x = db_mel_spec(x)

    with torch.autocast(device_type=cfg.device, enabled=cfg.amp):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    ema_model.update()
    scaler.update()
    scheduler.step()
    return loss


def train(
    model: nn.Module,
    ema_model: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: Config,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    eval_period: int,
    log_period: int,
    db_mel_spec: DbMelSpec,
    checkpoint_manager: CheckpointManager,
    start_step: int,
    metric_logger: JSONLMetricLogger,
    best_acc: float,
):
    model.train()

    it = iter(infinite_dataloader(train_loader))
    for step in range(start_step, cfg.max_steps + 1):
        train_loss = training_step(
            model, ema_model, optimizer, scaler, scheduler, it, cfg, db_mel_spec
        )

        if step % log_period == 0 or step == 1:
            print(f"{step}/{cfg.max_steps}: train_loss={train_loss.item():.4f}")
            metric_logger.log(
                {"split": "train", "loss": train_loss.item(), "step": step}
            )
        if step % eval_period == 0 or step == 1:
            val_loss, val_acc = evaluate(
                ema_model.ema_model, cfg, val_loader, db_mel_spec
            )
            print(
                f"{step}/{cfg.max_steps}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            metric_logger.log(
                {"split": "val", "loss": val_loss, "accuracy": val_acc, "step": step}
            )
            best = False
            if best_acc < val_acc:
                best = True
                best_acc = val_acc
            checkpoint_manager.save(
                ema_model.ema_model,
                step,
                best=best,
                best_acc=best_acc,
                optimizer=optimizer,
                scaler=scaler,
            )


@torch.no_grad()
def evaluate(model: nn.Module, cfg: Config, loader, db_mel_spec: DbMelSpec):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
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
