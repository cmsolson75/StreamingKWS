import torch
from .seed import seed_everything
from .train_utils import train, evaluate
from .dataloaders import load_dataloader
from .configs import Config
from .model import AudioClassifier
from .transforms import DbMelSpec
from .metric_logger import JSONLMetricLogger
import time
import argparse
import json
from .checkpoint_manager import CheckpointManager, RunManager

import subprocess
import platform
import getpass


def get_provenance(cfg: Config) -> dict:
    prov = {
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "git_dirty": subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout
        != "",
        "system": platform.system(),
        "machine": platform.machine(),
        "user": getpass.getuser(),
        "hostname": platform.node(),
        "seed": cfg.seed,
    }

    prov["accelerator"] = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available() and cfg.device == "cuda"
        else "MPS"
        if torch.backends.mps.is_available() and cfg.device == "mps"
        else "CPU"
    )

    return prov


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/config.yaml")
    parser.add_argument("--json_config", "-j", required=False)
    parser.add_argument(
        "--tags",
        "-t",
        nargs="+",
        metavar="TAG",
        help="List of tags (space-separated). E.g, --tags tag1 tag2 tag3",
        default="",
    )

    args, overwrites = parser.parse_known_args()
    if args.json_config is not None:
        cfg = Config.from_json(args.json_config).with_overrides(overwrites)
    else:
        cfg = Config.from_yaml(args.config).with_overrides(overwrites)

    run_manager = RunManager("runs", cfg, tags=args.tags, resume=cfg.resume)
    ckpt = CheckpointManager(run_manager)
    metrics_logger = JSONLMetricLogger(run_manager.path)

    with open(run_manager.path / "provenance.json", "w") as f:
        json.dump(get_provenance(cfg), f, indent=2)
    with open(run_manager.path / "overwrites.txt", "w") as f:
        for overwrite in overwrites:
            f.write(overwrite + "\n")
    print(f"Config: {cfg.model_dump_json(indent=4)}")

    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_loader = load_dataloader(cfg, "train")
    val_loader = load_dataloader(cfg, "val")

    db_mel_spec = DbMelSpec(cfg).to(cfg.device)
    model = AudioClassifier(len(cfg.subset)).to(cfg.device)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.amp)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    start_step, best_acc = ckpt.load(model, scaler=scaler, optimizer=optim)
    if start_step > 0:
        print(f"Resumed from step {start_step}")

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
        db_mel_spec,
        checkpoint_manager=ckpt,
        start_step=start_step,
        metric_logger=metrics_logger,
        best_acc=best_acc,
    )
    stop = time.perf_counter()
    print(f"Time: {stop - start:.4f} seconds")
    test_loader = load_dataloader(cfg, "test")
    print(evaluate(model, cfg, test_loader, db_mel_spec))


if __name__ == "__main__":
    launch()

# Speed: 281 seconds for 10k steps with default settings: LETS UP BATCH
