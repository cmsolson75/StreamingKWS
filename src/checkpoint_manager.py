from pathlib import Path
from safetensors.torch import save_file, load_file
from torch import nn
import torch
import torch.nn.functional as F
from .configs import Config
from datetime import datetime, timezone
import threading
import json
import subprocess


class RunManager:
    def __init__(
        self,
        base_path: str,
        cfg: Config,
        tags: list,
        resume: str | None = None,
    ):
        if resume:
            self.path = Path(base_path) / resume
            self.run_id = resume.split("+")[0]
        else:
            self.run_id = (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{cfg.hash()}"
            )
            slug = "".join(f"+{tag}" for tag in tags)
            self.path = Path(base_path) / f"{self.run_id}{slug}"
            self.path.mkdir(parents=True, exist_ok=True)
            cfg.to_json(self.path / "config.resolved.json")
        self.s3_bucket = cfg.remote_name if cfg.cloud_sync else None

        if self.s3_bucket:
            self._sync_pending = threading.Event()
            self._sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self._sync_thread.start()

    def _sync_worker(self):
        while True:
            self._sync_pending.wait()
            self._sync_pending.clear()
            try:
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "sync",
                        str(self.path),
                        f"s3://{self.s3_bucket}/{self.path.name}",
                    ],
                    check=True,
                    capture_output=True,
                )
                print(f"Synced to s3://{self.s3_bucket}/{self.path.name}")
            except Exception as e:
                print(f"S3 sync failed: {e}")

    def sync(self) -> None:
        if not self.s3_bucket:
            return
        self._sync_pending.set()


class CheckpointManager:
    def __init__(self, run_manager: RunManager):
        self.run_manager = run_manager
        self.ckpt_dir = self.run_manager.path / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self, model: nn.Module, step: int, best: bool, best_acc: float, **train_state
    ):
        step_dir = self.ckpt_dir / f"step_{step:06d}"
        step_dir.mkdir(exist_ok=True)

        model_path = step_dir / "model.safetensors"
        state_path = step_dir / "train_state.pt"

        save_file(model.state_dict(), model_path)
        state = {k: v.state_dict() for k, v in train_state.items()}
        state["best_acc"] = best_acc
        torch.save(state, state_path)

        # latest pointer
        (self.ckpt_dir / "latest.json").write_text(
            json.dumps({"step": step, "path": step_dir.name})
        )
        if best:
            # best pointer
            (self.ckpt_dir / "best.json").write_text(
                json.dumps({"step": step, "path": step_dir.name})
            )

        manifest = {
            "run_id": self.run_manager.run_id,
            "step": step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": [
                {"path": "model.safetensors", "bytes": model_path.stat().st_size},
                {"path": "train_state.pt", "bytes": state_path.stat().st_size},
            ],
        }
        (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        self.run_manager.sync()

    def load_latest(self) -> tuple[Path, int] | None:
        latest_file = self.ckpt_dir / "latest.json"
        if not latest_file.exists():
            return None
        latest = json.loads(latest_file.read_text())
        return self.ckpt_dir / latest["path"], latest["step"]

    def load(self, model: nn.Module, **train_state):
        result = self.load_latest()
        if result is None:
            return 0, 0.0
        step_dir, step = result
        state_dict = load_file(step_dir / "model.safetensors")
        model.load_state_dict(state_dict)

        saved_state = torch.load(step_dir / "train_state.pt", weights_only=False)
        for name, obj in train_state.items():
            if name in saved_state:
                obj.load_state_dict(saved_state[name])
        return step, saved_state["best_acc"]


if __name__ == "__main__":
    cfg = Config.from_yaml("configs/config.yaml")
    run_manager = RunManager(".", cfg, [], None, "20260212T214145Z_a9bbf6a412")

    ckpt_manager = CheckpointManager(run_manager)
    print(ckpt_manager.load_latest())
    model = nn.Linear(10, 10)
    optim = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: min((step + 1) / 10, 1)
    )
    ckpt_manager.save(model, 1, optim=optim, scheduler=scheduler)
    # x = torch.randn(1, 10)
    # y = torch.randint(0, 10, (1, 10), dtype=torch.float32)
    # epochs = 10
    # for epoch in range(epochs):
    #     logits = model(x)
    #     loss = F.cross_entropy(logits, y)

    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()
    #     scheduler.step()
    #     save_model_checkpoint(model)
    #     save_state(optim=optim, scheduler=scheduler)

    # print(optim.state_dict(), scheduler.state_dict())
