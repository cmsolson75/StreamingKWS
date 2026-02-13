from pathlib import Path
from safetensors.torch import save_file, load_file
from torch import nn
import torch
import torch.nn.functional as F
from .configs import Config
from datetime import datetime, timezone
import threading
from abc import ABC, abstractmethod
import json


class CloudSync(ABC):
    @abstractmethod
    def push(self, local_path: Path, remote_name: str) -> None: ...


class RunManager:
    def __init__(
        self,
        base_path: str,
        cfg: Config,
        tags: list,
        cloud_sync: CloudSync | None = None,
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
        self.cloud_sync = cloud_sync

        if self.cloud_sync:
            self._sync_pending = threading.Event()
            self._sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self._sync_thread.start

    def _sync_worker(self):
        while True:
            self._sync_pending.wait()
            self._sync_pending.clear()
            self.cloud_sync.push(self.path, self.path.name)

    def sync(self) -> None:
        if not self.cloud_sync:
            return
        self._sync_pending.set()


class CheckpointManager:

    def __init__(self, run_manager: RunManager):
        self.run_manager = run_manager
        self.ckpt_dir = self.run_manager.path / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, step: int, **train_state):
        step_dir = self.ckpt_dir / f"step_{step:06d}"
        step_dir.mkdir(exist_ok=True)

        model_path = step_dir / "model.safetensors"
        state_path = step_dir / "train_state.pt"

        save_file(model.state_dict(), model_path)
        torch.save({k: v.state_dict() for k, v in train_state.items()}, state_path)

        # latest pointer
        (self.ckpt_dir / "latest.json").write_text(
            json.dumps({"step": step, "path": step_dir.name})
        )

        manifest = {
            "run_id": self.run_manager.run_id,
            "step": step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": [
                {"path": "model.safetensors", "bytes": model_path.stat().st_size},
                {"path": "train_stats.pt", "bytes": state_path.stat().st_size},
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
