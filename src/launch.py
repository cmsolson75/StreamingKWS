import torch
from .seed import seed_everything
from .train import train, evaluate
from .dataloaders import load_dataloader
from .configs import Config
from .model import AudioClassifier
from .transforms import DbMelSpec
import time


def launch():
    seed_everything(42)
    cfg_path = "configs/config.yaml"
    cfg = Config.from_yaml(cfg_path)
    train_loader = load_dataloader(cfg, "train")
    val_loader = load_dataloader(cfg, "val")

    db_mel_spec = DbMelSpec(cfg).to(cfg.device)
    model = AudioClassifier(len(cfg.subset)).to(cfg.device)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.amp)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    start = time.perf_counter()
    train(
        model,
        optim,
        scaler,
        cfg,
        train_loader,
        val_loader,
        cfg.eval_period,
        cfg.log_period,
        db_mel_spec
    )
    stop = time.perf_counter()
    print(f"Time: {stop - start:.4f} seconds")
    test_loader = load_dataloader(cfg, "test")
    print(evaluate(model, cfg, test_loader, db_mel_spec))


if __name__ == "__main__":
    launch()

# Speed: 281 seconds for 10k steps with default settings: LETS UP BATCH